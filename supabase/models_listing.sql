-- What kind of furniture a model is (the catalogue's type filter) and whether shoppers see it (a vendor hides drafts
-- and test listings). Run after schema.sql. Safe to run more than once.

alter table public.models
  add column if not exists furniture_type text check (furniture_type in ('chairs', 'sofas', 'beds', 'tables', 'storage', 'doors', 'other')),
  add column if not exists listed boolean not null default true;

-- The type from the listing's name (English and Arabic words), when the vendor has not picked one.
create or replace function public.furniture_type_of(title text) returns text
  language sql immutable as $$
  select case
    when title ~* '(chair|stool|كرسي|كراسي|كرسى)' then 'chairs'
    when title ~* '(sofa|couch|كنبة|كنب|أريكة|اريكة|ركنة)' then 'sofas'
    when title ~* '(\mbed|bedroom|سرير)' then 'beds'
    when title ~* '(wardrobe|cupboard|closet|cabinet|خزانة|دولاب|نيش)' then 'storage'
    when title ~* '(table|desk|طاولة|ترابيزة|سفرة|مكتب)' then 'tables'
    when title ~* '(door|باب)' then 'doors'
    else 'other'
  end $$;

create or replace function public.models_fill_type() returns trigger
  language plpgsql as $$
begin
  if new.furniture_type is null then new.furniture_type := public.furniture_type_of(new.title); end if;
  return new;
end $$;

drop trigger if exists models_fill_type on public.models;
create trigger models_fill_type before insert or update of title, furniture_type on public.models
  for each row execute function public.models_fill_type();

update public.models set furniture_type = public.furniture_type_of(title) where furniture_type is null;

-- Vendors set both on their own models (models_own policy); the catalogue (backend) lists `listed` ones only.
grant update (furniture_type, listed) on public.models to authenticated;

notify pgrst, 'reload schema';
