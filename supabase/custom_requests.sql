-- Custom furniture requests without a 3D model, and what customers need to choose a vendor.
-- Builds on schema.sql, vendor_approval.sql, tasks.sql and marketplace.sql. Safe to run again.

-- Where a vendor delivers and installs (governorates / cities), and a short "about us" for their public page.
alter table public.vendors add column if not exists service_areas text[] not null default '{}';
alter table public.vendors add column if not exists bio text not null default '';
-- Photos of the vendor's own past work ([{url, caption}], at most 12), shown as "shared by the vendor", apart from
-- projects completed on Sha2ty.
alter table public.vendors add column if not exists portfolio jsonb not null default '[]'::jsonb;
do $$ begin
  alter table public.vendors add constraint vendors_portfolio_size check (jsonb_typeof(portfolio) = 'array' and jsonb_array_length(portfolio) <= 12);
exception when duplicate_object then null; end $$;
grant update (service_areas, bio, portfolio) on public.vendors to authenticated;

-- A listing's usual production time in days (null: not given; shown as "ask the vendor").
alter table public.models add column if not exists production_days integer
  check (production_days is null or production_days between 1 and 365);
grant update (production_days) on public.models to authenticated;
