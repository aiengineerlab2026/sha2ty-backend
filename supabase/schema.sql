-- Shatably vendor catalog. Run against the project Postgres database.

create extension if not exists "pgcrypto";

create table if not exists public.vendors (
  id uuid primary key references auth.users (id) on delete cascade,
  name text not null,
  location text not null,
  logo_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.models (
  id uuid primary key default gen_random_uuid(),
  vendor_id uuid not null references public.vendors (id) on delete cascade,
  title text not null default 'Untitled model',
  description text not null default '',
  width_m numeric,
  height_m numeric,
  depth_m numeric,
  source text not null check (source in ('photo', 'upload')),
  provider text check (provider in ('meshy', 'astra')),
  status text not null default 'draft' check (status in ('draft', 'generating', 'ready', 'failed')),
  glb_url text,
  source_image_url text,
  error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists models_vendor_id_created_at_idx
  on public.models (vendor_id, created_at desc);

create table if not exists public.model_variants (
  id uuid primary key default gen_random_uuid(),
  model_id uuid not null references public.models (id) on delete cascade,
  image_url text not null,
  sort_order int not null default 0
);

create index if not exists model_variants_model_id_idx
  on public.model_variants (model_id, sort_order);

alter table public.vendors enable row level security;
alter table public.models enable row level security;
alter table public.model_variants enable row level security;

grant usage on schema public to anon, authenticated;
grant select, insert, update, delete on public.vendors to authenticated;
grant select, insert, update, delete on public.models to authenticated;
grant select, insert, update, delete on public.model_variants to authenticated;

drop policy if exists vendors_own on public.vendors;
create policy vendors_own on public.vendors
  for all to authenticated
  using (id = auth.uid())
  with check (id = auth.uid());

drop policy if exists models_own on public.models;
create policy models_own on public.models
  for all to authenticated
  using (vendor_id = auth.uid())
  with check (vendor_id = auth.uid());

drop policy if exists variants_own on public.model_variants;
create policy variants_own on public.model_variants
  for all to authenticated
  using (
    exists (
      select 1 from public.models m
      where m.id = model_id and m.vendor_id = auth.uid()
    )
  )
  with check (
    exists (
      select 1 from public.models m
      where m.id = model_id and m.vendor_id = auth.uid()
    )
  );

insert into storage.buckets (id, name, public)
values ('cads', 'cads', true)
on conflict (id) do update set public = true;

drop policy if exists cads_select on storage.objects;
create policy cads_select on storage.objects
  for select to public
  using (bucket_id = 'cads');

drop policy if exists cads_insert on storage.objects;
create policy cads_insert on storage.objects
  for insert to authenticated
  with check (
    bucket_id = 'cads'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

drop policy if exists cads_update on storage.objects;
create policy cads_update on storage.objects
  for update to authenticated
  using (
    bucket_id = 'cads'
    and (storage.foldername(name))[1] = auth.uid()::text
  )
  with check (
    bucket_id = 'cads'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

drop policy if exists cads_delete on storage.objects;
create policy cads_delete on storage.objects
  for delete to authenticated
  using (
    bucket_id = 'cads'
    and (storage.foldername(name))[1] = auth.uid()::text
  );

create table if not exists public.floor_plans (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users (id) on delete cascade,
  name text not null default 'Untitled plan',
  source text not null default 'dimensions' check (source in ('image', 'dimensions')),
  source_image_url text,
  geometry jsonb not null default '{}'::jsonb,
  items jsonb not null default '[]'::jsonb,
  camera jsonb,
  warnings jsonb not null default '[]'::jsonb,
  glb_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists floor_plans_owner_id_updated_at_idx
  on public.floor_plans (owner_id, updated_at desc);

alter table public.floor_plans enable row level security;
grant select, insert, update, delete on public.floor_plans to authenticated;

drop policy if exists floor_plans_own on public.floor_plans;
create policy floor_plans_own on public.floor_plans
  for all to authenticated
  using (owner_id = auth.uid())
  with check (owner_id = auth.uid());

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


-- Prices for models, paints / floor finishes and carpets: a total price per item, or a price per m².
-- In a plan, paints are charged by the area they cover; models and carpets by the piece
-- (a per-m² model or carpet is charged for its own footprint). Safe to run more than once.
do $$
declare t text;
begin
  foreach t in array array['models', 'paints', 'carpets'] loop
    execute format('alter table public.%I add column if not exists price numeric check (price >= 0)', t);
    execute format('alter table public.%I add column if not exists price_unit text not null default %L', t, case when t = 'paints' then 'm2' else 'item' end);
    execute format('alter table public.%I drop constraint if exists %I', t, t || '_price_unit_check');
    execute format('alter table public.%I add constraint %I check (price_unit in (''item'', ''m2''))', t, t || '_price_unit_check');
  end loop;
end $$;


-- The catalogue (paints / floor finishes / carpets) is public: anyone can browse it, signed in or not.
-- Writes stay owner-only. Models are served publicly by the backend catalog API (ready models only).
grant select on public.paints, public.carpets to anon;

drop policy if exists paints_read on public.paints;
create policy paints_read on public.paints for select to anon, authenticated using (true);

drop policy if exists carpets_read on public.carpets;
create policy carpets_read on public.carpets for select to anon, authenticated using (true);


-- Texture tile size as width × height in metres (one image tile covers this much wall / floor).
-- Existing rows inherit the old square tile_m. Safe to run more than once.
alter table public.paints add column if not exists tile_width_m numeric check (tile_width_m > 0);
alter table public.paints add column if not exists tile_height_m numeric check (tile_height_m > 0);
update public.paints set tile_width_m = coalesce(tile_width_m, tile_m), tile_height_m = coalesce(tile_height_m, tile_m)
  where tile_width_m is null or tile_height_m is null;


-- Model category: furniture, tables, wardrobes, wall art / images, TV & AV equipment, lights, ceiling lights (mounted at the ceiling in plans),
-- air conditioners. Existing models stay furniture. Safe to run more than once.
alter table public.models add column if not exists category text not null default 'furniture';
alter table public.models drop constraint if exists models_category_check;
alter table public.models add constraint models_category_check
  check (category in ('furniture', 'table', 'wardrobe', 'tv', 'lighting', 'ceiling_light', 'air_conditioner', 'wall_art'));


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


-- L-shaped corner units (cupboards, wardrobes, kitchen runs) stand along two walls: width_m is the length
-- along the first wall, length2_m along the second, depth_m how deep each arm is. Safe to run more than once.
alter table public.models add column if not exists shape text not null default 'straight';
alter table public.models add column if not exists length2_m numeric check (length2_m > 0);
alter table public.models drop constraint if exists models_shape_check;
alter table public.models add constraint models_shape_check check (shape in ('straight', 'l_left', 'l_right'));


-- 1) Variant images for paints, carpets and windows (models keep model_variants).
-- 2) Reviews: 1–5 stars and a comment per signed-in user per catalogue item.
-- Both are public to read; only owners write. Safe to run more than once.

create table if not exists public.item_variants (
  id uuid primary key default gen_random_uuid(),
  item_kind text not null check (item_kind in ('paint', 'carpet', 'window')),
  item_id uuid not null,
  image_url text not null,
  sort_order int not null default 0,
  created_at timestamptz not null default now()
);
create index if not exists item_variants_item_idx on public.item_variants (item_kind, item_id, sort_order);

alter table public.item_variants enable row level security;
grant select on public.item_variants to anon;
grant select, insert, update, delete on public.item_variants to authenticated;

drop policy if exists item_variants_read on public.item_variants;
create policy item_variants_read on public.item_variants for select to anon, authenticated using (true);

-- Writes: only the vendor who owns the paint / carpet / window.
drop policy if exists item_variants_own on public.item_variants;
create policy item_variants_own on public.item_variants
  for all to authenticated
  using (
    (item_kind = 'paint' and exists (select 1 from public.paints p where p.id = item_id and p.vendor_id = auth.uid()))
    or (item_kind = 'carpet' and exists (select 1 from public.carpets c where c.id = item_id and c.vendor_id = auth.uid()))
    or (item_kind = 'window' and exists (select 1 from public.windows w where w.id = item_id and w.vendor_id = auth.uid()))
  )
  with check (
    (item_kind = 'paint' and exists (select 1 from public.paints p where p.id = item_id and p.vendor_id = auth.uid()))
    or (item_kind = 'carpet' and exists (select 1 from public.carpets c where c.id = item_id and c.vendor_id = auth.uid()))
    or (item_kind = 'window' and exists (select 1 from public.windows w where w.id = item_id and w.vendor_id = auth.uid()))
  );

-- Model variants are shown publicly on catalogue pages too.
grant select on public.model_variants to anon;
drop policy if exists variants_read on public.model_variants;
create policy variants_read on public.model_variants for select to anon, authenticated using (true);

create table if not exists public.reviews (
  id uuid primary key default gen_random_uuid(),
  item_kind text not null check (item_kind in ('model', 'paint', 'carpet', 'window')),
  item_id uuid not null,
  user_id uuid not null references auth.users (id) on delete cascade default auth.uid(),
  author_name text not null default '' check (char_length(author_name) <= 80),
  rating int not null check (rating between 1 and 5),
  comment text not null default '' check (char_length(comment) <= 2000),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (item_kind, item_id, user_id)
);
create index if not exists reviews_item_idx on public.reviews (item_kind, item_id, created_at desc);

alter table public.reviews enable row level security;
grant select on public.reviews to anon;
grant select, insert, update, delete on public.reviews to authenticated;

drop policy if exists reviews_read on public.reviews;
create policy reviews_read on public.reviews for select to anon, authenticated using (true);

drop policy if exists reviews_own on public.reviews;
create policy reviews_own on public.reviews
  for all to authenticated
  using (user_id = auth.uid())
  with check (user_id = auth.uid());


notify pgrst, 'reload schema';

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
