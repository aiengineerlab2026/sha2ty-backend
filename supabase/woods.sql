-- Default woods: the platform's catalogue of wood materials (oak … laminate), owned by the Sha2ty vendor and
-- editable only by admins (through the admin studio, as that vendor). Each vendor ticks which of them they work
-- with (vendor_woods); a vendor's own model offers its customers those woods only. Run after fabrics.sql and
-- platform_vendor.sql. Safe to run more than once.

create or replace function public.is_platform_vendor(v uuid) returns boolean
  language sql stable security definer set search_path = public
  as $$ select exists (select 1 from public.vendors where id = v and platform) $$;
grant execute on function public.is_platform_vendor(uuid) to anon, authenticated;

-- Vendors keep their own fabrics; wood rows are the platform's only.
drop policy if exists fabrics_own on public.fabrics;
create policy fabrics_own on public.fabrics
  for all to authenticated
  using (vendor_id = auth.uid() and (kind = 'fabric' or public.is_platform_vendor(auth.uid())))
  with check (vendor_id = auth.uid() and public.vendor_approved(auth.uid()) and (kind = 'fabric' or public.is_platform_vendor(auth.uid())));

create table if not exists public.vendor_woods (
  vendor_id uuid not null references public.vendors (id) on delete cascade,
  wood_id uuid not null references public.fabrics (id) on delete cascade,
  created_at timestamptz not null default now(),
  primary key (vendor_id, wood_id)
);
create index if not exists vendor_woods_wood_idx on public.vendor_woods (wood_id);

create or replace function public.is_platform_wood(w uuid) returns boolean
  language sql stable security definer set search_path = public
  as $$ select exists (select 1 from public.fabrics f join public.vendors v on v.id = f.vendor_id where f.id = w and f.kind = 'wood' and v.platform) $$;
grant execute on function public.is_platform_wood(uuid) to anon, authenticated;

alter table public.vendor_woods enable row level security;
grant select on public.vendor_woods to anon;
grant select, insert, delete on public.vendor_woods to authenticated;

drop policy if exists vendor_woods_read on public.vendor_woods;
create policy vendor_woods_read on public.vendor_woods for select to anon, authenticated using (true);

drop policy if exists vendor_woods_own on public.vendor_woods;
create policy vendor_woods_own on public.vendor_woods
  for all to authenticated
  using (vendor_id = auth.uid())
  with check (vendor_id = auth.uid() and public.vendor_approved(auth.uid()) and public.is_platform_wood(wood_id));

-- The woods a model can be made in, chosen by its vendor (or admins for Sha2ty's models) from the woods they work
-- with. Null = all of them (also ones added later).
alter table public.models add column if not exists woods uuid[] check (array_length(woods, 1) <= 100);

-- The defaults (images in frontend/public/woods, seamless; admins can replace them and change prices). Prices in EGP.
insert into public.fabrics (vendor_id, kind, name, wood_type, description, texture_url, tile_m, price, price_unit, thicknesses_mm, sheet_w_m, sheet_h_m, roughness)
select v.id, 'wood', d.name, d.wood_type, d.description, '/woods/' || d.wood_type || '.jpg', d.tile_m, d.price, d.price_unit, d.thicknesses_mm, d.sheet_w, d.sheet_h, d.roughness
from public.vendors v
cross join (values
  ('Oak', 'oak', 'Hard, strong solid wood with a clear open grain. Frames, legs and tables.', 0.6, 90000, 'm3', '{20,25,40,50}'::numeric[], null::numeric, null::numeric, 0.55),
  ('Beech', 'beech', 'Even, fine-grained solid wood; takes stain well. Chairs, frames and legs.', 0.6, 55000, 'm3', '{20,25,40,50}', null, null, 0.55),
  ('Pine', 'pine', 'Light, affordable softwood with bold grain. Slats, frames and painted pieces.', 0.6, 25000, 'm3', '{18,25,40}', null, null, 0.6),
  ('Walnut', 'walnut', 'Rich dark solid hardwood. Premium headboards, tables and visible parts.', 0.6, 140000, 'm3', '{20,25,40}', null, null, 0.5),
  ('Teak', 'teak', 'Durable, oily hardwood that resists moisture. Outdoor and premium furniture.', 0.6, 160000, 'm3', '{20,25,40}', null, null, 0.5),
  ('Ash', 'ash', 'Pale, tough hardwood with a strong grain. Bent parts, frames and legs.', 0.6, 70000, 'm3', '{20,25,40}', null, null, 0.55),
  ('Mahogany', 'mahogany', 'Reddish-brown hardwood with a fine straight grain. Classic furniture.', 0.6, 110000, 'm3', '{20,25,40}', null, null, 0.5),
  ('MDF', 'mdf', 'Smooth engineered board for painted or veneered panels, headboards and cabinets.', 1.22, 1400, 'sheet', '{6,12,16,18,25}', 1.22, 2.44, 0.75),
  ('HDF', 'hdf', 'Thin, dense fibreboard for backs, drawer bottoms and door skins.', 1.22, 450, 'sheet', '{3,4,6}', 1.22, 2.44, 0.75),
  ('Plywood', 'plywood', 'Cross-laminated layers: strong and stable. Bed bases, frames and carcasses.', 1.22, 2200, 'sheet', '{9,12,15,18}', 1.22, 2.44, 0.65),
  ('Particleboard', 'particleboard', 'Economical chipboard for hidden or laminated panels.', 1.22, 900, 'sheet', '{16,18}', 1.83, 2.44, 0.8),
  ('Blockboard', 'blockboard', 'Solid-wood core between veneers: rigid long panels and doors.', 1.22, 1800, 'sheet', '{18,25}', 1.22, 2.44, 0.65),
  ('Melamine', 'melamine', 'Hard-wearing decorative surface pressed onto board. Easy-clean panels.', 0.6, 250, 'm2', null, null, null, 0.4),
  ('Walnut veneer', 'veneer', 'A thin slice of real walnut on a board: the look of solid wood for less.', 0.6, 350, 'm2', '{0.6}', null, null, 0.5),
  ('Laminate / HPL', 'laminate', 'High-pressure laminate: very durable surface for tops and fronts.', 0.6, 450, 'm2', '{0.8,1}', null, null, 0.3)
) as d(name, wood_type, description, tile_m, price, price_unit, thicknesses_mm, sheet_w, sheet_h, roughness)
where v.platform
  and not exists (select 1 from public.fabrics f where f.vendor_id = v.id and f.kind = 'wood' and f.wood_type = d.wood_type);

notify pgrst, 'reload schema';
