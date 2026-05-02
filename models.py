from pydantic import BaseModel, Field, field_validator
from typing import Optional


class InvoiceItem(BaseModel):
    deskripsi_item: str = Field(..., description="Deskripsi / nama item")
    jumlah_item: Optional[float] = Field(None, description="Kuantitas / jumlah item")
    harga_satuan: Optional[float] = Field(None, description="Harga per satuan")
    total_item: Optional[float] = Field(None, description="Total harga item (jumlah x harga satuan)")
    id_ref_pengeluaran: Optional[str] = Field(
        None,
        description="Kode subkelompok pengeluaran 6 digit, e.g. '010101' untuk Beras"
    )

    @field_validator("id_ref_pengeluaran", mode="before")
    @classmethod
    def coerce_ref_to_string(cls, v):
        if v is None:
            return None
        s = str(int(v)) if isinstance(v, (int, float)) else str(v).strip()
        return s.zfill(6)


class InvoiceResponse(BaseModel):
    nama_toko: Optional[str] = Field(None, description="Nama toko / vendor / penerbit")
    tanggal_bukti: Optional[str] = Field(None, description="Tanggal transaksi / bukti (ISO 8601 jika memungkinkan)")
    nomor_bukti: Optional[str] = Field(None, description="Nomor invoice / faktur / struk")
    nilai_total: Optional[float] = Field(None, description="Total nilai transaksi (grand total)")
    items: list[InvoiceItem] = []
    vendor_address: Optional[str] = Field(None, description="Alamat vendor")
    due_date: Optional[str] = Field(None, description="Tanggal jatuh tempo pembayaran")
    billing_address: Optional[str] = Field(None, description="Alamat tagihan / pembeli")
    payment_terms: Optional[str] = Field(None, description="Termin pembayaran, e.g. NET30")
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    discount: Optional[float] = None
    currency: Optional[str] = Field(None, description="Kode mata uang ISO 4217, e.g. IDR, USD")
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_text_hint: Optional[str] = Field(None, description="Catatan jika ekstraksi sulit atau tidak yakin")


class ExtractionError(BaseModel):
    detail: str
    raw_response: Optional[str] = None
