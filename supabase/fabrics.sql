-- Upholstery fabrics: vendors upload a tiling fabric image and its real tile size; shoppers pick them as the
-- texture of a model part in the customise editor (the only texture source there). Safe to run more than once.
create table if not exists public.fabrics (
  id uuid primary key default gen_random_uuid(),
  vendor_id uuid not null references public.vendors (id) on delete cascade,
  name text not null check (length(name) between 1 and 120),
  texture_url text not null,
  -- Real-world size in metres of one texture tile.
  tile_m numeric not null default 0.3 check (tile_m > 0),
  -- Price per running metre of fabric; null = price on request.
  price_per_m numeric check (price_per_m >= 0),
  created_at timestamptz not null default now()
);

alter table public.fabrics add column if not exists price_per_m numeric check (price_per_m >= 0);

create index if not exists fabrics_vendor_id_idx on public.fabrics (vendor_id, created_at desc);

alter table public.fabrics enable row level security;
grant select on public.fabrics to anon;
grant select, insert, update, delete on public.fabrics to authenticated;

drop policy if exists fabrics_read on public.fabrics;
create policy fabrics_read on public.fabrics for select to anon, authenticated using (true);

-- Only approved vendors add fabrics, and only as themselves.
drop policy if exists fabrics_own on public.fabrics;
create policy fabrics_own on public.fabrics
  for all to authenticated
  using (vendor_id = auth.uid())
  with check (vendor_id = auth.uid() and public.vendor_approved(auth.uid()));

notify pgrst, 'reload schema';

-- Wood materials share this library (one picker, one pricing path in the editor): kind = 'wood', the wood type
-- (oak … laminate, frontend/lib/designs/materials.ts WOOD_TYPES), what it is priced by, the thicknesses offered,
-- the sheet size for sheet goods, and optional PBR maps. Fabrics keep kind 'fabric' and price_per_m.
alter table public.fabrics add column if not exists kind text not null default 'fabric';
alter table public.fabrics drop constraint if exists fabrics_kind_check;
alter table public.fabrics add constraint fabrics_kind_check check (kind in ('fabric', 'wood'));
alter table public.fabrics add column if not exists wood_type text check (wood_type ~ '^[a-z]{2,20}$');
alter table public.fabrics add column if not exists description text check (length(description) <= 500);
alter table public.fabrics add column if not exists price numeric check (price >= 0);
alter table public.fabrics add column if not exists price_unit text not null default 'm';
alter table public.fabrics drop constraint if exists fabrics_price_unit_check;
alter table public.fabrics add constraint fabrics_price_unit_check check (price_unit in ('m', 'm2', 'm3', 'sheet'));
alter table public.fabrics add column if not exists thicknesses_mm numeric[] check (array_length(thicknesses_mm, 1) <= 20);
alter table public.fabrics add column if not exists sheet_w_m numeric check (sheet_w_m > 0);
alter table public.fabrics add column if not exists sheet_h_m numeric check (sheet_h_m > 0);
alter table public.fabrics add column if not exists normal_url text;
alter table public.fabrics add column if not exists roughness numeric check (roughness between 0 and 1);
alter table public.fabrics drop constraint if exists fabrics_wood_check;
alter table public.fabrics add constraint fabrics_wood_check check (kind <> 'wood' or wood_type is not null);

notify pgrst, 'reload schema';
