-- Vendor windows: an image (background removed) shown on a wall face at its real width × height.
-- Public read like the rest of the catalogue; writes are owner-only. Safe to run more than once.
create table if not exists public.windows (
  id uuid primary key default gen_random_uuid(),
  vendor_id uuid not null references public.vendors (id) on delete cascade,
  name text not null,
  width_m numeric not null check (width_m > 0),
  height_m numeric not null check (height_m > 0),
  image_url text not null,
  price numeric check (price >= 0),
  price_unit text not null default 'item' check (price_unit in ('item', 'm2')),
  created_at timestamptz not null default now()
);

create index if not exists windows_vendor_id_idx on public.windows (vendor_id, created_at desc);

alter table public.windows enable row level security;
grant select on public.windows to anon;
grant select, insert, update, delete on public.windows to authenticated;

drop policy if exists windows_read on public.windows;
create policy windows_read on public.windows for select to anon, authenticated using (true);

drop policy if exists windows_own on public.windows;
create policy windows_own on public.windows
  for all to authenticated
  using (vendor_id = auth.uid())
  with check (vendor_id = auth.uid());

notify pgrst, 'reload schema';
