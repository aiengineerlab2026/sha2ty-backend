-- 3D is optional: vendors can list a model with photos only (source 'none', glb_url null).
-- Photo-only models show in the shop but not in the plan builder (it needs a GLB).
alter table public.models drop constraint if exists models_source_check;
alter table public.models add constraint models_source_check check (source in ('photo', 'upload', 'none'));

notify pgrst, 'reload schema';
