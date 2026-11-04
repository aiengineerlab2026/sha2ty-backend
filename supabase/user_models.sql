-- Customers' private models: a photo of their own furniture turned into 3D in the app (same generation as vendors,
-- counted against the same per-account tries). Only the owner sees or changes them; they are never in the catalogue
-- and are used in the owner's own planner and AR. Files live in the cads bucket under the owner's folder.
-- Safe to run more than once.

create table if not exists public.user_models (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null default auth.uid() references auth.users (id) on delete cascade,
  title text not null default '' check (length(title) <= 200),
  description text not null default '' check (length(description) <= 4000),
  category text not null default 'furniture' check (category ~ '^[a-z_]{2,40}$'),
  width_m numeric check (width_m > 0 and width_m <= 50),
  height_m numeric check (height_m > 0 and height_m <= 50),
  depth_m numeric check (depth_m > 0 and depth_m <= 50),
  glb_url text,
  source_glb_url text,
  source_image_url text,
  components jsonb,
  status text not null default 'generating' check (status in ('generating', 'draft', 'ready', 'failed')),
  error text,
  created_at timestamptz not null default now()
);

create index if not exists user_models_owner_idx on public.user_models (owner_id, created_at desc);

alter table public.user_models enable row level security;
grant select, insert, update, delete on public.user_models to authenticated;

drop policy if exists user_models_own on public.user_models;
create policy user_models_own on public.user_models
  for all to authenticated
  using (owner_id = auth.uid())
  with check (owner_id = auth.uid());

notify pgrst, 'reload schema';
