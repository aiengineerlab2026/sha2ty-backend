-- Model category: furniture, tables, wardrobes, wall art / images, TV & AV equipment, lights, ceiling lights (mounted at the ceiling in plans),
-- air conditioners. Existing models stay furniture. Safe to run more than once.
alter table public.models add column if not exists category text not null default 'furniture';
alter table public.models drop constraint if exists models_category_check;
alter table public.models add constraint models_category_check
  check (category in ('furniture', 'table', 'wardrobe', 'tv', 'lighting', 'ceiling_light', 'air_conditioner', 'wall_art'));

notify pgrst, 'reload schema';
