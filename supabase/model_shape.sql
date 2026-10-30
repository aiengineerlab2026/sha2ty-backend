-- L-shaped corner units (cupboards, wardrobes, kitchen runs) stand along two walls: width_m is the length
-- along the first wall, length2_m along the second, depth_m how deep each arm is. Safe to run more than once.
alter table public.models add column if not exists shape text not null default 'straight';
alter table public.models add column if not exists length2_m numeric check (length2_m > 0);
alter table public.models drop constraint if exists models_shape_check;
alter table public.models add constraint models_shape_check check (shape in ('straight', 'l_left', 'l_right'));

notify pgrst, 'reload schema';
