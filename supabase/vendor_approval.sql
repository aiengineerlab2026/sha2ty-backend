-- Vendor approval: new vendor signups wait for the admin (backend /api/admin) before they sell.
-- Existing vendors are approved by this migration; new rows default to pending.

alter table public.vendors
  add column if not exists status text not null default 'approved' check (status in ('pending', 'approved', 'rejected')),
  add column if not exists full_name text,
  add column if not exists email text,
  add column if not exists lat double precision,
  add column if not exists lng double precision;
alter table public.vendors alter column status set default 'approved';  -- vendor_auto_approve.sql

-- Vendors edit their own profile but can never set their own status (only the service role can).
revoke insert, update on public.vendors from authenticated;
grant insert (id, name, location, logo_url, full_name, email, lat, lng) on public.vendors to authenticated;
grant update (name, location, logo_url, lat, lng, updated_at) on public.vendors to authenticated;

create or replace function public.vendor_approved(v uuid) returns boolean
  language sql stable security definer set search_path = public
  as $$ select exists (select 1 from public.vendors where id = v and status = 'approved') $$;
grant execute on function public.vendor_approved(uuid) to anon, authenticated;

-- The public catalogue lists approved vendors only (owners still see their own rows via *_own).
drop policy if exists paints_read on public.paints;
create policy paints_read on public.paints for select to anon, authenticated using (public.vendor_approved(vendor_id));
drop policy if exists carpets_read on public.carpets;
create policy carpets_read on public.carpets for select to anon, authenticated using (public.vendor_approved(vendor_id));
drop policy if exists windows_read on public.windows;
create policy windows_read on public.windows for select to anon, authenticated using (public.vendor_approved(vendor_id));

notify pgrst, 'reload schema';
