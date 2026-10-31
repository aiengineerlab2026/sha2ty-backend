-- Vendors are approved on signup (no admin review); the admin can still suspend/reject later.
alter table public.vendors alter column status set default 'approved';
update public.vendors set status = 'approved' where status = 'pending';
