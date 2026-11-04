-- Implementation requests from customised designs, bid revisions, task chat and vendor reviews.
-- Builds on tasks.sql (freelance_tasks, task_bids, accept_task_bid, advance_task) and designs.sql.
-- Safe to run more than once.

-- A request made from a design points at the exact (immutable) version vendors bid on.
alter table public.freelance_tasks add column if not exists design_id uuid references public.designs (id) on delete set null;
alter table public.freelance_tasks add column if not exists design_version_id uuid references public.design_versions (id) on delete set null;
alter table public.freelance_tasks add column if not exists deadline date;
create index if not exists freelance_tasks_design_version on public.freelance_tasks (design_version_id) where design_version_id is not null;

-- Who may bid on a request. "open": every approved vendor. "vendor_exclusive": only exclusive_vendor_id (the vendor
-- whose catalogue model the design started from). Written by the backend from the design's frozen source; a
-- missing vendor never turns an exclusive request into an open one (nobody else can see it).
alter table public.freelance_tasks add column if not exists audience text not null default 'open';
alter table public.freelance_tasks add column if not exists exclusive_vendor_id uuid references public.vendors (id) on delete set null;
do $$ begin
  alter table public.freelance_tasks add constraint freelance_tasks_audience check (audience in ('open', 'vendor_exclusive'));
exception when duplicate_object then null;
end $$;

-- Lookups that read tasks and bids without row-level security. tasks.sql's policies referred to each other
-- (tasks → bids → tasks) and Postgres stops that with "infinite recursion detected in policy" (bids failed).
-- The policies below use these instead; each answers only for the given ids.
create or replace function public.task_customer(task uuid) returns uuid
  language sql stable security definer set search_path = public as $$ select customer_id from public.freelance_tasks where id = task $$;
create or replace function public.task_status(task uuid) returns text
  language sql stable security definer set search_path = public as $$ select status from public.freelance_tasks where id = task $$;
create or replace function public.has_bid(task uuid, vendor uuid) returns boolean
  language sql stable security definer set search_path = public as $$ select exists (select 1 from public.task_bids where task_id = task and vendor_id = vendor) $$;
create or replace function public.hired_vendor(task uuid) returns uuid
  language sql stable security definer set search_path = public as $$
    select b.vendor_id from public.freelance_tasks t join public.task_bids b on b.id = t.awarded_bid_id where t.id = task $$;
-- The vendor may take part in this request: an open one, or the exclusive request addressed to them.
create or replace function public.task_open_to(task uuid, vendor uuid) returns boolean
  language sql stable security definer set search_path = public as $$
    select exists (select 1 from public.freelance_tasks where id = task and (audience = 'open' or exclusive_vendor_id = vendor)) $$;
grant execute on function public.task_customer(uuid), public.task_status(uuid), public.has_bid(uuid, uuid), public.hired_vendor(uuid), public.task_open_to(uuid, uuid) to authenticated;

drop policy if exists tasks_read on public.freelance_tasks;
create policy tasks_read on public.freelance_tasks for select to authenticated using (
  customer_id = auth.uid()
  or (status = 'published' and public.vendor_approved(auth.uid()) and (audience = 'open' or exclusive_vendor_id = auth.uid()))
  or public.has_bid(id, auth.uid())
);

-- Customers publish plan requests themselves (open, no design); requests from a design are made by the backend,
-- which sets the audience from the design's origin. So a client cannot publish a design request, or pick its audience.
drop policy if exists tasks_insert on public.freelance_tasks;
create policy tasks_insert on public.freelance_tasks for insert to authenticated with check (
  customer_id = auth.uid() and status = 'published' and audience = 'open' and exclusive_vendor_id is null
  and design_id is null and design_version_id is null
);
drop policy if exists bids_read on public.task_bids;
create policy bids_read on public.task_bids for select to authenticated using (
  vendor_id = auth.uid() or public.task_customer(task_id) = auth.uid()
);
drop policy if exists bids_insert on public.task_bids;
create policy bids_insert on public.task_bids for insert to authenticated with check (
  vendor_id = auth.uid() and status = 'submitted' and public.vendor_approved(auth.uid()) and public.task_status(task_id) = 'published'
  and public.task_open_to(task_id, auth.uid())
);


-- Bids: a vendor revises (price, days, note) or withdraws their own bid while the task is open, and may submit
-- again after withdrawing. Accepting / rejecting stays with accept_task_bid.
alter table public.task_bids add column if not exists updated_at timestamptz not null default now();
grant update (price, days, note, status, updated_at) on public.task_bids to authenticated;
drop policy if exists bids_update_own on public.task_bids;
create policy bids_update_own on public.task_bids for update to authenticated
  using (vendor_id = auth.uid() and status in ('submitted', 'withdrawn'))
  with check (
    vendor_id = auth.uid()
    and status in ('submitted', 'withdrawn')
    and public.task_status(task_id) = 'published'
    and public.task_open_to(task_id, auth.uid())
  );

-- Chat: one conversation per task and vendor, between that vendor and the task's customer only.
create table if not exists public.task_messages (
  id uuid primary key default gen_random_uuid(),
  task_id uuid not null references public.freelance_tasks (id) on delete cascade,
  vendor_id uuid not null references public.vendors (id) on delete cascade,
  sender_id uuid not null references auth.users (id) on delete cascade default auth.uid(),
  body text not null default '' check (char_length(body) <= 4000),
  attachment_url text check (attachment_url is null or char_length(attachment_url) <= 1000),
  created_at timestamptz not null default now(),
  read_at timestamptz,
  check (char_length(body) > 0 or attachment_url is not null)
);
create index if not exists task_messages_conversation on public.task_messages (task_id, vendor_id, created_at);

-- The vendor already wrote in this task's thread (a question before bidding): the customer may answer.
create or replace function public.vendor_wrote(task uuid, vendor uuid) returns boolean
  language sql stable security definer set search_path = public as $$
    select exists (select 1 from public.task_messages where task_id = task and vendor_id = vendor and sender_id = vendor) $$;
grant execute on function public.vendor_wrote(uuid, uuid) to authenticated;

alter table public.task_messages enable row level security;
grant select, insert on public.task_messages to authenticated;
grant update (read_at) on public.task_messages to authenticated;

drop policy if exists task_messages_read on public.task_messages;
create policy task_messages_read on public.task_messages for select to authenticated using (
  vendor_id = auth.uid() or public.task_customer(task_id) = auth.uid()
);

-- The customer writes to vendors who bid (or already talked to them); a vendor writes about a task they bid on,
-- or asks about an open task before bidding.
drop policy if exists task_messages_send on public.task_messages;
create policy task_messages_send on public.task_messages for insert to authenticated with check (
  sender_id = auth.uid()
  and (
    (public.task_customer(task_id) = auth.uid() and (public.has_bid(task_id, vendor_id) or public.vendor_wrote(task_id, vendor_id)))
    or (
      vendor_id = auth.uid()
      and (public.has_bid(task_id, auth.uid())
           or (public.task_status(task_id) = 'published' and public.vendor_approved(auth.uid()) and public.task_open_to(task_id, auth.uid())))
    )
  )
);

-- Only the recipient marks a message read.
drop policy if exists task_messages_read_mark on public.task_messages;
create policy task_messages_read_mark on public.task_messages for update to authenticated
  using (
    sender_id <> auth.uid()
    and (vendor_id = auth.uid() or public.task_customer(task_id) = auth.uid())
  )
  with check (sender_id <> auth.uid());

-- New messages reach open chats live (Realtime applies the policies above per subscriber).
do $$ begin
  if exists (select 1 from pg_publication where pubname = 'supabase_realtime')
     and not exists (select 1 from pg_publication_tables where pubname = 'supabase_realtime' and tablename = 'task_messages') then
    alter publication supabase_realtime add table public.task_messages;
  end if;
end $$;

-- Reviews: the customer of a completed task reviews the vendor they hired, once. Public to read; nobody edits
-- them afterwards (vendors cannot change a customer's review).
create table if not exists public.vendor_reviews (
  id uuid primary key default gen_random_uuid(),
  task_id uuid not null unique references public.freelance_tasks (id) on delete cascade,
  vendor_id uuid not null references public.vendors (id) on delete cascade,
  customer_id uuid not null references auth.users (id) on delete cascade default auth.uid(),
  rating int not null check (rating between 1 and 5),
  comment text not null default '' check (char_length(comment) <= 2000),
  created_at timestamptz not null default now()
);
create index if not exists vendor_reviews_vendor on public.vendor_reviews (vendor_id, created_at desc);

alter table public.vendor_reviews enable row level security;
grant select on public.vendor_reviews to anon, authenticated;
grant insert on public.vendor_reviews to authenticated;

drop policy if exists vendor_reviews_read on public.vendor_reviews;
create policy vendor_reviews_read on public.vendor_reviews for select to anon, authenticated using (true);

drop policy if exists vendor_reviews_insert on public.vendor_reviews;
create policy vendor_reviews_insert on public.vendor_reviews for insert to authenticated with check (
  customer_id = auth.uid()
  and public.task_customer(task_id) = auth.uid()
  and public.task_status(task_id) = 'completed'
  and public.hired_vendor(task_id) = vendor_id
);

notify pgrst, 'reload schema';
