-- Vendor wall paints: a flat colour or a tiling texture. Any signed-in user can use them in a plan;
-- only the owning vendor can change them. Texture images live in the cads bucket under the vendor's folder.
create table if not exists public.paints (
  id uuid primary key default gen_random_uuid(),
  vendor_id uuid not null references public.vendors (id) on delete cascade,
  name text not null,
  kind text not null check (kind in ('color', 'texture')),
  -- Paint colour, or for textures the image's average colour (used by the GLB export).
  color text not null check (color ~ '^#[0-9a-fA-F]{6}$'),
  texture_url text,
  -- Real-world size in metres of one texture tile.
  tile_m numeric not null default 1 check (tile_m > 0),
  created_at timestamptz not null default now(),
  check (kind = 'color' or texture_url is not null)
);

create index if not exists paints_vendor_id_idx on public.paints (vendor_id, created_at desc);

alter table public.paints enable row level security;
grant select, insert, update, delete on public.paints to authenticated;

drop policy if exists paints_read on public.paints;
create policy paints_read on public.paints for select to authenticated using (true);

drop policy if exists paints_own on public.paints;
create policy paints_own on public.paints
  for all to authenticated
  using (vendor_id = auth.uid())
  with check (vendor_id = auth.uid());

notify pgrst, 'reload schema';
