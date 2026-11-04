-- Vendors' proposed changes to a requested model (backend/app/api/proposals.py). Before bidding, a vendor may edit
-- the customer's model (size, parts, materials) and send it with a note; the customer accepts it (it becomes a new
-- version of their design and the request's specification) or rejects it. Written only by the backend (service
-- key); the vendor and the customer read their own. Run after tasks.sql and marketplace.sql. Safe to run again.

create table if not exists public.task_proposals (
  id uuid primary key default gen_random_uuid(),
  task_id uuid not null references public.freelance_tasks (id) on delete cascade,
  vendor_id uuid not null references public.vendors (id) on delete cascade,
  -- The design version it was made from, and the proposed item (same shape as design_versions.data).
  base_version_id uuid,
  data jsonb not null,
  glb_url text,
  thumbnail_url text,
  note text not null default '' check (length(note) <= 2000),
  status text not null default 'pending' check (status in ('pending', 'accepted', 'rejected')),
  created_at timestamptz not null default now(),
  decided_at timestamptz
);

create index if not exists task_proposals_task_idx on public.task_proposals (task_id, created_at desc);

alter table public.task_proposals enable row level security;
grant select on public.task_proposals to authenticated;

drop policy if exists task_proposals_read on public.task_proposals;
create policy task_proposals_read on public.task_proposals for select to authenticated using (
  vendor_id = auth.uid() or public.task_customer(task_id) = auth.uid()
);

notify pgrst, 'reload schema';
