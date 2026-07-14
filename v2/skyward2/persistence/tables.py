from piccolo.columns import JSONB, Float, Integer, Timestamptz, Varchar
from piccolo.table import Table


class ProviderRow(Table, tablename="providers"):
    """A registered provider account.

    ``name`` is the alias, ``kind`` picks the adapter. Two rows may share a kind:
    two AWS accounts, two Vast keys, two regions. That is why the compute refers
    to a provider id, not to a kind.

    ``credentials`` holds the secret in the clear for now. It never leaves this
    table: no read path selects it, and the API never returns it.
    """

    id = Varchar(primary_key=True)
    name = Varchar(unique=True, index=True)
    kind = Varchar(index=True)
    credentials = JSONB(default={})
    config = JSONB(default={})
    created_at = Timestamptz()
    offers_fetched_at = Timestamptz(null=True, default=None)
    last_error = Varchar(null=True, default=None)


class OfferRow(Table, tablename="offers"):
    """A cached offer.

    A cache, not a ledger: a refresh replaces a provider's rows wholesale,
    because an offer that vanished from the catalog must vanish here — keeping
    it would let a compute be planned against hardware that no longer exists.

    ``expires_at`` comes from the provider's own TTL. A marketplace expires in
    minutes; a fixed fleet can hold for hours.
    """

    id = Varchar(primary_key=True)
    offer_id = Varchar(index=True)
    provider_id = Varchar(index=True)
    provider_name = Varchar()
    kind = Varchar(index=True)
    instance_type = Varchar()
    accelerator = Varchar(null=True, default=None, index=True)
    accelerator_count = Integer(default=0, index=True)
    vram = Float(null=True, default=None, index=True)
    cpus = Integer(default=0)
    memory_gb = Float(default=0.0)
    disk_gb = Float(null=True, default=None)
    region = Varchar(null=True, default=None)
    spot_price = Float(null=True, default=None)
    on_demand_price = Float(null=True, default=None)
    price = Float(null=True, default=None, index=True)
    available = Integer(null=True, default=None)
    specific = JSONB(default={})
    fetched_at = Timestamptz()
    expires_at = Timestamptz(index=True)


TABLES = (ProviderRow, OfferRow)
