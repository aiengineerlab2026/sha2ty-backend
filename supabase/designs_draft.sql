-- A design saved without a name (a temporary one is given): shown as a draft until the customer names it, e.g.
-- when publishing a request. Run after designs.sql. Safe to run more than once.
alter table public.designs add column if not exists draft boolean not null default false;
notify pgrst, 'reload schema';
