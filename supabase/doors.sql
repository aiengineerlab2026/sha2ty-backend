-- Vendor doors: a front-on photo (background removed) at its real width × height in metres.
-- Shoppers fit them into the plan's door openings only; the plan builder checks the size fits.
create table if not exists public.doors (
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

create index if not exists doors_vendor_id_idx on public.doors (vendor_id, created_at desc);

alter table public.doors enable row level security;
grant select on public.doors to anon;
grant select, insert, update, delete on public.doors to authenticated;

-- Public read for approved vendors (vendor_approval.sql), owners manage their own.
drop policy if exists doors_read on public.doors;
create policy doors_read on public.doors for select to anon, authenticated using (public.vendor_approved(vendor_id));
drop policy if exists doors_own on public.doors;
create policy doors_own on public.doors for all to authenticated using (vendor_id = auth.uid()) with check (vendor_id = auth.uid());

-- Doors have variant photos and reviews like windows.
alter table public.item_variants drop constraint if exists item_variants_item_kind_check;
alter table public.item_variants add constraint item_variants_item_kind_check check (item_kind in ('paint', 'carpet', 'window', 'door'));
alter table public.reviews drop constraint if exists reviews_item_kind_check;
alter table public.reviews add constraint reviews_item_kind_check check (item_kind in ('model', 'paint', 'carpet', 'window', 'door'));

notify pgrst, 'reload schema';
