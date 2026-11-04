-- Customer designs: one customised catalogue item, one room, or a whole apartment, with an immutable version
-- history. `data` is the shared design document (backend/app/schemas/design.py, frontend/lib/designs/schema.ts,
-- mobile/src/lib/designs.ts). A version is never changed: saving, restoring and duplicating all add versions.
-- Catalogue originals are referenced (source_model_id, and a snapshot in data), never modified.
-- Writes go through the backend (validated, owner checked) via add_design_version; owners can read their rows.
-- Safe to run more than once. Needs schema.sql (models, floor_plans).

create table if not exists public.designs (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users (id) on delete cascade,
  kind text not null check (kind in ('item', 'room', 'apartment')),
  title text not null default '' check (char_length(title) <= 200),
  source_model_id uuid references public.models (id) on delete set null, -- item: the catalogue model it started from
  plan_id uuid references public.floor_plans (id) on delete set null,    -- room / apartment: the planner's working copy
  current_version_id uuid,
  version_count integer not null default 0 check (version_count >= 0),
  thumbnail_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  deleted_at timestamptz -- soft delete: versions stay for anything that references them (published tasks)
);

create table if not exists public.design_versions (
  id uuid primary key default gen_random_uuid(),
  design_id uuid not null references public.designs (id) on delete cascade,
  number integer not null check (number > 0),
  schema text not null default 'sha2ty.design/1',
  data jsonb not null check (jsonb_typeof(data) = 'object'),
  glb_url text,       -- item: the customised GLB of this version (null: the catalogue original, edits replayed)
  thumbnail_url text,
  note text not null default '' check (char_length(note) <= 500),
  restored_from uuid references public.design_versions (id) on delete set null,
  client_token uuid,  -- idempotency: a retried save returns the version it already made
  created_by uuid not null references auth.users (id) on delete cascade,
  created_at timestamptz not null default now(),
  unique (design_id, number),
  unique (design_id, client_token)
);

do $$ begin
  alter table public.designs add constraint designs_current_version_fk
    foreign key (current_version_id) references public.design_versions (id) on delete set null deferrable initially deferred;
exception when duplicate_object then null;
end $$;

-- Idempotent first save: the same token from the same owner returns the design already made (double click).
alter table public.designs add column if not exists client_token uuid;
create unique index if not exists designs_owner_client_token on public.designs (owner_id, client_token) where client_token is not null;

create index if not exists designs_owner_updated on public.designs (owner_id, updated_at desc) where deleted_at is null;
create index if not exists design_versions_design on public.design_versions (design_id, number desc);

-- Versions are immutable.
create or replace function public.design_versions_immutable() returns trigger
  language plpgsql as $$
begin
  raise exception 'Design versions cannot be changed';
end $$;
drop trigger if exists design_versions_no_update on public.design_versions;
create trigger design_versions_no_update before update on public.design_versions
  for each row execute function public.design_versions_immutable();

-- The only way a version is added: locks the design, numbers the version, makes it current. Owner checked here
-- as well as in the backend. Idempotent on (design, client_token).
create or replace function public.add_design_version(
  p_design uuid, p_owner uuid, p_data jsonb, p_glb_url text, p_thumbnail_url text, p_note text,
  p_restored_from uuid, p_client_token uuid
) returns public.design_versions
  language plpgsql security definer set search_path = public as $$
declare
  d public.designs;
  v public.design_versions;
begin
  select * into d from public.designs where id = p_design for update;
  if d.id is null or d.owner_id <> p_owner or d.deleted_at is not null then
    raise exception 'Design not found' using errcode = 'P0002';
  end if;
  if p_client_token is not null then
    select * into v from public.design_versions where design_id = p_design and client_token = p_client_token;
    if v.id is not null then return v; end if;
  end if;
  if p_restored_from is not null and not exists (
    select 1 from public.design_versions where id = p_restored_from and design_id = p_design
  ) then
    raise exception 'Version not found' using errcode = 'P0002';
  end if;
  insert into public.design_versions (design_id, number, schema, data, glb_url, thumbnail_url, note, restored_from, client_token, created_by)
  values (p_design, d.version_count + 1, coalesce(p_data ->> 'schema', 'sha2ty.design/1'), p_data, p_glb_url,
          p_thumbnail_url, coalesce(p_note, ''), p_restored_from, p_client_token, p_owner)
  returning * into v;
  update public.designs
     set current_version_id = v.id, version_count = v.number, updated_at = now(),
         thumbnail_url = coalesce(p_thumbnail_url, thumbnail_url)
   where id = p_design;
  return v;
end $$;

revoke all on function public.add_design_version(uuid, uuid, jsonb, text, text, text, uuid, uuid) from public, anon, authenticated;
grant execute on function public.add_design_version(uuid, uuid, jsonb, text, text, text, uuid, uuid) to service_role;

alter table public.designs enable row level security;
alter table public.design_versions enable row level security;

-- Owners read their own designs and versions (mobile and web may read directly); every write is the backend's.
revoke insert, update, delete on public.designs from anon, authenticated;
revoke insert, update, delete on public.design_versions from anon, authenticated;
grant select on public.designs to authenticated;
grant select on public.design_versions to authenticated;

drop policy if exists designs_owner_read on public.designs;
create policy designs_owner_read on public.designs for select to authenticated
  using (owner_id = auth.uid() and deleted_at is null);

drop policy if exists design_versions_owner_read on public.design_versions;
create policy design_versions_owner_read on public.design_versions for select to authenticated
  using (exists (select 1 from public.designs d where d.id = design_id and d.owner_id = auth.uid() and d.deleted_at is null));

notify pgrst, 'reload schema';
