import json
from openai import OpenAI
from pydantic import ValidationError
from models import InvoiceResponse, InvoiceItem

SYSTEM_PROMPT = (
    "You are an invoice and receipt data extraction engine. "
    "Extract structured data from the provided image(s) and return ONLY valid JSON — no prose, no markdown. "
    "If a field is not present or unclear, set it to null. "
    "For confidence, return a float between 0.0 and 1.0 reflecting image quality and extraction completeness."
)

REF_PENGELUARAN = """\
Gunakan tabel referensi berikut untuk mengisi id_ref_pengeluaran pada setiap item.
Pilih KODE SUBKELOMPOK PENGELUARAN (6 digit) yang paling sesuai dengan deskripsi item.
Jika tidak ada yang cocok, set null.

KODE   | NAMA SUBKELOMPOK
010101 | Beras
010102 | Jagung
010103 | Ubi Jalar
010104 | Talas
010105 | Kentang
010106 | Mie
010107 | Tepung Terigu
010108 | Tepung Beras
010109 | Tepung Tapioka
010110 | Sagu
010111 | Sorgum
010112 | Gandum
010113 | Singkong
010199 | Karbohidrat Lainya
010201 | Telur Ayam
010202 | Telur Bebek
010203 | Ayam
010204 | Daging Sapi
010205 | Daging Kerbau
010206 | Ikan Kembung
010207 | Ikan Tongkol
010208 | Ikan Cakalang
010209 | Ikan Tuna
010210 | Ikan Patin
010211 | Ikan Lele
010212 | Ikan Nila
010213 | Ikan Mas
010214 | Ikan Mujair
010215 | Udang
010216 | Teri
010217 | Telor Puyuh
010218 | Telor Asin
010219 | Belut
010299 | Protein Hewani Lainya
010301 | Tempe
010302 | Tahu
010303 | Kacang Hijau
010304 | Kacang Merah
010305 | Kacang Kedelai
010306 | Kacang Tanah
010307 | Kacang Polong
010399 | Protein Nabati Lainya
010401 | Bayam
010402 | Kangkung
010403 | Sawi Hijau
010404 | Sawi Putih
010405 | Selada
010406 | Daun Singkong
010407 | Daun Kelor
010408 | Daun Katuk
010409 | Labu Siam
010410 | Terong Ungu
010411 | Wortel
010412 | Ketimun
010413 | Tomat
010414 | Kol
010415 | Kembang Kol
010416 | Brokoli
010417 | Buncis
010418 | Kacang Panjang
010419 | Kacang Kapri
010420 | Baligo/Kundur
010421 | Lobak
010422 | Daun Bawang
010423 | Tauge
010424 | Jamur
010425 | Rumput Laut
010426 | Pokcoy
010427 | Seledri
010499 | Sayur Lainya
010501 | Pisang
010502 | Apel
010503 | Jeruk
010504 | Pepaya
010505 | Anggur
010506 | Semangka
010507 | Melon
010508 | Kelengkeng
010509 | Salak
010510 | Matoa
010511 | Mangga
010512 | Manggis
010513 | Rambutan
010599 | Buah-buahan Lainya
010601 | Minyak Goreng
010602 | Kecap
010603 | Saos
010604 | Garam
010605 | Gula
010606 | Penyedap Rasa
010607 | Bawang Putih
"""

EXTRACTION_PROMPT = """\
You are extracting data from an invoice/receipt image. Follow these steps IN ORDER:

## STEP 1 — RAW TRANSCRIPTION (fill "_scratch" field first)
Before filling any structured field, transcribe EVERY item row exactly as written.
Format each line as: "ROW N: [raw qty+unit] | [raw item name] | [raw unit price] | [raw total]"
Example: "ROW 1: 3.600 pcs | Roti | 3.000 | 10.800.000"
- Read left to right, top to bottom
- Do NOT skip any row
- Write exactly what you see, even if illegible — use "???" for unreadable parts

## STEP 2 — PARSE INTO JSON (after completing _scratch)
Use your _scratch transcription to fill the structured fields below.
Return this exact JSON (all fields required, use null if absent):
{
  "_scratch": "ROW 1: ... | ROW 2: ... (your full transcription here)",
  "nama_toko": "string or null",
  "tanggal_bukti": "YYYY-MM-DD or original string or null",
  "nomor_bukti": "string or null",
  "nilai_total": number or null,
  "items": [
    {
      "deskripsi_item": "string",
      "jumlah_item": number or null,
      "harga_satuan": number or null,
      "total_item": number or null,
      "id_ref_pengeluaran": "MUST be a quoted string e.g. \"010101\", never a number, or null"
    }
  ],
  "vendor_address": "string or null",
  "due_date": "YYYY-MM-DD or original string or null",
  "billing_address": "string or null",
  "payment_terms": "string or null",
  "subtotal": number or null,
  "tax": number or null,
  "discount": number or null,
  "currency": "ISO 4217 code or null",
  "confidence": <see rubric below>,
  "raw_text_hint": "catatan singkat dalam bahasa Indonesia yang natural, hanya jika ada kendala — atau null. Contoh yang baik: 'Tulisan tangan kurang jelas di beberapa baris item.', 'Nomor faktur tidak tertera pada dokumen.', 'Beberapa angka harga sulit dibedakan karena tinta buram.', 'Stempel menutupi sebagian total akhir.' Jangan tulis dalam bahasa Inggris."
}

## COLUMN RULES
Most invoices: [Qty+Unit] [Item Name] [Unit Price] [Total Price]
- LAST numeric column = total_item
- SECOND-TO-LAST numeric column = harga_satuan
- Verify: jumlah_item × harga_satuan ≈ total_item — if math fails, re-read that row from the image
- Sum of all total_item ≈ nilai_total

## NUMBER FORMAT
Indonesian: dots are thousand separators → 10.800.000 = 10800000, 3.000 = 3000
Common digit confusions in handwriting: 1↔7, 3↔8, 6↔0, 2↔7

## CONFIDENCE RUBRIC
- Printed receipt, all fields clear → 0.85–1.0
- Printed receipt, 1-2 fields ambiguous → 0.70–0.84
- Handwritten invoice, math validates, most fields clear → 0.55–0.69
- Handwritten invoice, several fields ambiguous → 0.40–0.54
- Heavy distortion or many unreadable fields → 0.20–0.39
Handwritten documents should NEVER exceed 0.70.

""" + REF_PENGELUARAN


def _validate_math(result: InvoiceResponse) -> InvoiceResponse:
    """
    Post-processing: flag rows where jumlah × harga_satuan doesn't match total_item,
    and flag if sum of items doesn't match nilai_total. Appends warnings to raw_text_hint
    and lowers confidence accordingly.
    """
    warnings: list[str] = []
    bad_rows = 0

    for i, item in enumerate(result.items):
        if item.jumlah_item and item.harga_satuan and item.total_item:
            computed = round(item.jumlah_item * item.harga_satuan)
            actual = round(item.total_item)
            tolerance = actual * 0.05
            if abs(computed - actual) > max(tolerance, 1):
                warnings.append(
                    f"baris {i+1} ({item.deskripsi_item}): "
                    f"{item.jumlah_item} × {item.harga_satuan:,.0f} = {computed:,.0f}, "
                    f"tapi total tercatat {actual:,.0f}"
                )
                bad_rows += 1

    if result.nilai_total and result.items:
        item_totals = [it.total_item for it in result.items if it.total_item is not None]
        if item_totals:
            computed_total = round(sum(item_totals))
            actual_total = round(result.nilai_total)
            tolerance = actual_total * 0.02
            if abs(computed_total - actual_total) > max(tolerance, 1):
                warnings.append(
                    f"jumlah semua item ({computed_total:,}) tidak cocok dengan total faktur ({actual_total:,})"
                )

    if warnings:
        existing_hint = result.raw_text_hint or ""
        mismatch_note = "Terdeteksi ketidaksesuaian perhitungan: " + "; ".join(warnings) + "."
        combined = existing_hint + " " + mismatch_note if existing_hint else mismatch_note
        result.raw_text_hint = combined

        penalty = min(bad_rows * 0.05 + (0.05 if len(warnings) > bad_rows else 0), 0.25)
        result.confidence = round(max(0.1, result.confidence - penalty), 2)

    return result


def extract_invoice_data(client: OpenAI, image_blocks: list[dict]) -> InvoiceResponse:
    content: list[dict] = []

    for block in image_blocks:
        b64_data = block["source"]["data"]
        media_type = block["source"]["media_type"]
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{b64_data}",
                "detail": "high",
            }
        })

    content.append({"type": "text", "text": EXTRACTION_PROMPT})

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )

    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}\nRaw: {raw[:300]}")

    try:
        items = [InvoiceItem.model_validate(item) for item in data.get("items", [])]
    except ValidationError as e:
        raise ValueError(f"Invalid item structure from model: {e}")

    try:
        invoice = InvoiceResponse(
            nama_toko=data.get("nama_toko"),
            tanggal_bukti=data.get("tanggal_bukti"),
            nomor_bukti=data.get("nomor_bukti"),
            nilai_total=data.get("nilai_total"),
            items=items,
            vendor_address=data.get("vendor_address"),
            due_date=data.get("due_date"),
            billing_address=data.get("billing_address"),
            payment_terms=data.get("payment_terms"),
            subtotal=data.get("subtotal"),
            tax=data.get("tax"),
            discount=data.get("discount"),
            currency=data.get("currency"),
            confidence=data.get("confidence", 0.5),
            raw_text_hint=data.get("raw_text_hint"),
        )
    except ValidationError as e:
        raise ValueError(f"Response validation failed: {e}")

    return _validate_math(invoice)
