-- A bid's terms as separate fields the customer compares side by side (price basis, quantity, wood / material,
-- fabric, installation, warranty, payment). `note` stays for anything else. Run after marketplace.sql. Safe to run
-- more than once. Older bids keep their text in `note` and read as a total price.

alter table public.task_bids
  add column if not exists price_basis text not null default 'total' check (price_basis in ('total', 'per_item')),
  add column if not exists quantity integer check (quantity is null or (quantity > 0 and quantity <= 1000)),
  add column if not exists material text not null default '' check (char_length(material) <= 300),
  add column if not exists fabric text not null default '' check (char_length(fabric) <= 300),
  add column if not exists installation boolean,
  add column if not exists warranty text not null default '' check (char_length(warranty) <= 300),
  add column if not exists payment_terms text not null default '' check (char_length(payment_terms) <= 300);

-- Vendors revise these with the rest of their bid while the request is open (policy bids_update_own).
grant update (price, days, note, status, updated_at, price_basis, quantity, material, fabric, installation, warranty, payment_terms)
  on public.task_bids to authenticated;

notify pgrst, 'reload schema';
