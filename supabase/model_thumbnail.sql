-- A render of the saved 3D model (thumbnail_url) and the save it came from (version: glb_url and
-- thumbnail_url are written together). source_image_url stays the vendor's original photo. Safe to run more than once.
alter table public.models add column if not exists thumbnail_url text;
alter table public.models add column if not exists version integer not null default 1;

notify pgrst, 'reload schema';
