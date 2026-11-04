-- Furniture paints. Admins keep the colour palette (paint_colors, written by backend/app/api/admin_paint_colors.py
-- with the service key); customers paint model parts in those colours only. Vendors add paint products in a palette
-- colour to the part-material library (fabrics, kind = 'paint'), priced per kg or litre with how far one unit goes,
-- so customers see how much paint a part takes. Run after fabrics.sql and woods.sql. Safe to run more than once.

create table if not exists public.paint_colors (
  id uuid primary key default gen_random_uuid(),
  name text not null check (length(name) between 1 and 80),
  hex text not null check (hex ~ '^#[0-9a-fA-F]{6}$'),
  created_at timestamptz not null default now()
);

alter table public.paint_colors enable row level security;
grant select on public.paint_colors to anon, authenticated;
drop policy if exists paint_colors_read on public.paint_colors;
create policy paint_colors_read on public.paint_colors for select to anon, authenticated using (true);

-- Paint rows have no texture: their look is the palette colour.
alter table public.fabrics alter column texture_url drop not null;
alter table public.fabrics drop constraint if exists fabrics_kind_check;
alter table public.fabrics add constraint fabrics_kind_check check (kind in ('fabric', 'wood', 'paint'));
alter table public.fabrics drop constraint if exists fabrics_price_unit_check;
alter table public.fabrics add constraint fabrics_price_unit_check check (price_unit in ('m', 'm2', 'm3', 'sheet', 'kg', 'l'));
alter table public.fabrics add column if not exists paint_color_id uuid references public.paint_colors (id) on delete cascade;
-- Square metres one kg or litre covers in one coat; unset = 10 (frontend/lib/designs/materials.ts quote).
alter table public.fabrics add column if not exists coverage_m2 numeric check (coverage_m2 > 0 and coverage_m2 <= 100);
alter table public.fabrics drop constraint if exists fabrics_paint_check;
alter table public.fabrics add constraint fabrics_paint_check check (
  kind <> 'paint' or (paint_color_id is not null and price_unit in ('kg', 'l'))
);
-- Only a wood row needs a wood type. fabrics.sql was written when the only other kind was fabric.
alter table public.fabrics drop constraint if exists fabrics_wood_check;
alter table public.fabrics add constraint fabrics_wood_check check (kind <> 'wood' or wood_type is not null);
alter table public.fabrics drop constraint if exists fabrics_texture_check;
alter table public.fabrics add constraint fabrics_texture_check check (kind = 'paint' or texture_url is not null);

-- Vendors keep their own fabrics and paints; wood rows are the platform's only.
drop policy if exists fabrics_own on public.fabrics;
create policy fabrics_own on public.fabrics
  for all to authenticated
  using (vendor_id = auth.uid() and (kind in ('fabric', 'paint') or public.is_platform_vendor(auth.uid())))
  with check (vendor_id = auth.uid() and public.vendor_approved(auth.uid()) and (kind in ('fabric', 'paint') or public.is_platform_vendor(auth.uid())));

-- A starting palette (admins change it in the admin portal).
insert into public.paint_colors (name, hex)
select d.name, d.hex
from (values
  ('White', '#f5f5f2'), ('Off white', '#ece6d8'), ('Light grey', '#c9c9c4'), ('Charcoal', '#3b3d40'), ('Black', '#1d1d1f'),
  ('Beige', '#d8c3a0'), ('Walnut brown', '#5b3a24'), ('Navy', '#1f2f4d'), ('Sage green', '#9caf88'), ('Terracotta', '#b5583a')
) as d (name, hex)
where not exists (select 1 from public.paint_colors);

notify pgrst, 'reload schema';
