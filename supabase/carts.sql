-- One cart per signed-in user (guests keep theirs in the browser until they log in).
create table if not exists public.carts (
  user_id uuid primary key references auth.users (id) on delete cascade default auth.uid(),
  lines jsonb not null default '[]'::jsonb,
  updated_at timestamptz not null default now()
);

alter table public.carts enable row level security;

drop policy if exists "carts: owner reads" on public.carts;
create policy "carts: owner reads" on public.carts for select using (auth.uid() = user_id);
drop policy if exists "carts: owner inserts" on public.carts;
create policy "carts: owner inserts" on public.carts for insert with check (auth.uid() = user_id);
drop policy if exists "carts: owner updates" on public.carts;
create policy "carts: owner updates" on public.carts for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
drop policy if exists "carts: owner deletes" on public.carts;
create policy "carts: owner deletes" on public.carts for delete using (auth.uid() = user_id);
