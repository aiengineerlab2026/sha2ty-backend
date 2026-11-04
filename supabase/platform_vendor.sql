-- The platform's own catalogue: one vendor row, "Sha2ty", marked platform = true, owned by a system user that
-- only admins enter (backend/app/api/admin.py studio-session). Its models are ordinary models of that vendor,
-- so the public catalogue, the planner and every vendor workflow treat them like any other listing.
-- Only the server (service role) can mark or unmark the platform vendor. Safe to run more than once.

alter table public.vendors add column if not exists platform boolean not null default false;
create unique index if not exists vendors_one_platform on public.vendors (platform) where platform;

create or replace function public.vendors_platform_guard() returns trigger
  language plpgsql as $$
begin
  if coalesce(auth.role(), '') <> 'service_role'
     and ((tg_op = 'INSERT' and new.platform) or (tg_op = 'UPDATE' and new.platform is distinct from old.platform)) then
    raise exception 'Only the platform can mark its own vendor';
  end if;
  return new;
end $$;

drop trigger if exists vendors_platform_guard on public.vendors;
create trigger vendors_platform_guard before insert or update on public.vendors
  for each row execute function public.vendors_platform_guard();

notify pgrst, 'reload schema';
