-- A saved plan offered to vendors. spec is the snapshot they bid on (quantities and prices at publish time).
-- Status changes go through the functions below; row policies do not allow rewriting status from the client.

create table if not exists public.freelance_tasks (
  id uuid primary key default gen_random_uuid(),
  customer_id uuid not null references auth.users (id) on delete cascade default auth.uid(),
  plan_id uuid references public.floor_plans (id) on delete set null,
  title text not null,
  scope text not null default 'apartment' check (scope in ('item', 'room', 'apartment')),
  service text not null default 'complete' check (service in ('furniture', 'upholstery', 'painting', 'flooring', 'complete')),
  description text not null default '',
  location text not null default '',
  budget_max numeric check (budget_max is null or budget_max >= 0),
  status text not null default 'published' check (status in ('published', 'awarded', 'in_progress', 'completed', 'cancelled')),
  spec jsonb not null default '{}'::jsonb,
  awarded_bid_id uuid,
  created_at timestamptz not null default now()
);

create table if not exists public.task_bids (
  id uuid primary key default gen_random_uuid(),
  task_id uuid not null references public.freelance_tasks (id) on delete cascade,
  vendor_id uuid not null references public.vendors (id) on delete cascade,
  price numeric not null check (price >= 0),
  days integer not null check (days > 0),
  note text not null default '',
  status text not null default 'submitted' check (status in ('submitted', 'withdrawn', 'accepted', 'rejected')),
  created_at timestamptz not null default now(),
  unique (task_id, vendor_id)
);

alter table public.freelance_tasks enable row level security;
alter table public.task_bids enable row level security;

grant select, insert on public.freelance_tasks to authenticated;
grant select, insert on public.task_bids to authenticated;

drop policy if exists tasks_read on public.freelance_tasks;
create policy tasks_read on public.freelance_tasks for select to authenticated using (
  customer_id = auth.uid()
  or (status = 'published' and public.vendor_approved(auth.uid()))
  or exists (select 1 from public.task_bids b where b.task_id = id and b.vendor_id = auth.uid())
);

drop policy if exists tasks_insert on public.freelance_tasks;
create policy tasks_insert on public.freelance_tasks for insert to authenticated
  with check (customer_id = auth.uid() and status = 'published');

drop policy if exists bids_read on public.task_bids;
create policy bids_read on public.task_bids for select to authenticated using (
  vendor_id = auth.uid()
  or exists (select 1 from public.freelance_tasks t where t.id = task_id and t.customer_id = auth.uid())
);

drop policy if exists bids_insert on public.task_bids;
create policy bids_insert on public.task_bids for insert to authenticated
  with check (
    vendor_id = auth.uid()
    and status = 'submitted'
    and public.vendor_approved(auth.uid())
    and exists (select 1 from public.freelance_tasks t where t.id = task_id and t.status = 'published')
  );

create or replace function public.accept_task_bid(bid uuid) returns void
  language plpgsql security definer set search_path = public as $$
declare
  t uuid;
begin
  select task_id into t from public.task_bids where id = bid and status = 'submitted';
  if t is null then raise exception 'Bid is not open'; end if;
  if not exists (select 1 from public.freelance_tasks where id = t and customer_id = auth.uid() and status = 'published') then
    raise exception 'Not an open task of yours';
  end if;
  update public.task_bids set status = 'rejected' where task_id = t and id <> bid and status = 'submitted';
  update public.task_bids set status = 'accepted' where id = bid;
  update public.freelance_tasks set status = 'awarded', awarded_bid_id = bid where id = t;
end $$;

create or replace function public.advance_task(task uuid, next text) returns void
  language plpgsql security definer set search_path = public as $$
declare
  row public.freelance_tasks;
  vendor uuid;
begin
  select * into row from public.freelance_tasks where id = task;
  if row.id is null then raise exception 'Task not found'; end if;
  select vendor_id into vendor from public.task_bids where id = row.awarded_bid_id;
  if next = 'cancelled' and row.customer_id = auth.uid() and row.status = 'published' then
    update public.freelance_tasks set status = 'cancelled' where id = task;
  elsif next = 'in_progress' and vendor = auth.uid() and row.status = 'awarded' then
    update public.freelance_tasks set status = 'in_progress' where id = task;
  elsif next = 'completed' and row.customer_id = auth.uid() and row.status = 'in_progress' then
    update public.freelance_tasks set status = 'completed' where id = task;
  else
    raise exception 'That status change is not allowed';
  end if;
end $$;

revoke all on function public.accept_task_bid(uuid) from public;
revoke all on function public.advance_task(uuid, text) from public;
grant execute on function public.accept_task_bid(uuid) to authenticated;
grant execute on function public.advance_task(uuid, text) to authenticated;

notify pgrst, 'reload schema';
