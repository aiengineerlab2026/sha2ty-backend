-- Orders: what a signed-in user bought (a snapshot of their cart lines at checkout).
-- The "Bought items" page lists them; Match can reuse those items without re-measuring.
create table if not exists public.orders (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade default auth.uid(),
  lines jsonb not null check (jsonb_typeof(lines) = 'array' and jsonb_array_length(lines) > 0),
  total numeric not null default 0,
  created_at timestamptz not null default now()
);
create index if not exists orders_user_created on public.orders (user_id, created_at desc);

alter table public.orders enable row level security;
-- Owners read and place their own orders; nobody edits or deletes one afterwards.
drop policy if exists "orders: owner reads" on public.orders;
create policy "orders: owner reads" on public.orders for select using (auth.uid() = user_id);
drop policy if exists "orders: owner inserts" on public.orders;
create policy "orders: owner inserts" on public.orders for insert with check (auth.uid() = user_id);

notify pgrst, 'reload schema';
