-- Paints can be for walls or floors; vendors can also offer carpets (an image laid on the floor at a real size).
-- Run after paints.sql. Safe to run more than once.
alter table public.paints add column if not exists surface text not null default 'wall';
alter table public.paints drop constraint if exists paints_surface_check;
alter table public.paints add constraint paints_surface_check check (surface in ('wall', 'floor'));

create table if not exists public.carpets (
  id uuid primary key default gen_random_uuid(),
  vendor_id uuid not null references public.vendors (id) on delete cascade,
  name text not null,
  width_m numeric not null check (width_m > 0),
  length_m numeric not null check (length_m > 0),
  image_url text not null,
  created_at timestamptz not null default now()
);

create index if not exists carpets_vendor_id_idx on public.carpets (vendor_id, created_at desc);

alter table public.carpets enable row level security;
grant select, insert, update, delete on public.carpets to authenticated;

drop policy if exists carpets_read on public.carpets;
create policy carpets_read on public.carpets for select to authenticated using (true);

drop policy if exists carpets_own on public.carpets;
create policy carpets_own on public.carpets
  for all to authenticated
  using (vendor_id = auth.uid())
  with check (vendor_id = auth.uid());

notify pgrst, 'reload schema';
