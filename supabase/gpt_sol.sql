-- GPT Sol as a third 3D generator (admin console → 3D generator). Widens the provider checks. Safe to run more than once.

alter table public.generation_usage drop constraint if exists generation_usage_provider_check;
alter table public.generation_usage add constraint generation_usage_provider_check check (provider in ('meshy', 'astra', 'sol'));

alter table public.models drop constraint if exists models_provider_check;
alter table public.models add constraint models_provider_check check (provider in ('meshy', 'astra', 'sol'));

notify pgrst, 'reload schema';
