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
