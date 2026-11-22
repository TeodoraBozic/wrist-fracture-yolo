# Image i Fracture EDA

[02 — Image EDA](../notebooks/02_image_eda.ipynb) pokušava da dekodira sve jednoznačne
slikovne identifikatore iz četiri originalna ZIP-a. [03 — Fracture EDA](../notebooks/03_fracture_eda.ipynb)
koristi isti tehnički keš i samostalno računa lokalni kontrast. Uputstvo za pokretanje
je u [README-u](../README.md).

## Tehničke mere

- Jedna slika se dekodira u datom trenutku, sa originalnom preciznošću. Bit depth
  PNG-a potiče iz IHDR zaglavlja, a mode, dtype i kanali iz dekodirane slike.
  Praznine u metrikama i zabeleženi problemi imaju zasebne imenioce.
- Statistike intenziteta koriste originalni DN. Kontrast je P99–P1. Full-scale
  mere koriste teorijski raspon potvrđene unsigned dubine. Varijansa Laplasijana
  na izvornom rasteru je proxy oštrine koji zavisi i od rezolucije.
- Galerije koriste P1–P99 granice isključivo za prikaz. Tehnički uzorak pokriva
  projekcije, proizvođače i uzrasne raspone. Odvojeno se prikazuju tehnički ekstremi.

## Fracture mere

- Koordinate se pretvaraju u pikselne ivice formulom `x0=(cx−w/2)W`,
  `x1=(cx+w/2)W` i analogno po y. Overlay koristi kontinualne ivice slike.
  Klase se razlikuju bojom i originalnim nazivom.
- Lokalni kontrast je P95–P5 u box-u i okolnom prstenu, u DN jedinicama.
  Floor/ceil uključuju presečene piksele. Prsten dodaje 25% odgovarajuće
  dimenzije box-a po strani, najmanje jedan piksel, i ograničen je ivicom slike.
  Prazan prsten i greške se beleže. Susedne anotacije mogu biti deo prstena.
- Slike sa praznom anotacijom, slike sa drugim klasama i nepoznat status
  prikazuju se odvojeno. Slike bez fracture box-a se zadržavaju. Prazan
  `fracture_visible` prikazuje se kao neupisan.
- Rubni primeri su kandidati za ručnu proveru. Njihove posledice po detekciju
  mogu se utvrditi u kasnijim eksperimentima.

## Nastavak rada i rezultati

Potpis tehničkog keša obuhvata ZIP inventar (putanja, veličina, mtime i CRC člana),
implementaciju, relevantne verzije biblioteka i podešavanja. Fracture keš dodatno
koristi otisak box koordinata. JSON serije od po 128 slika objavljuju se po završetku,
uz proveru ključeva, broja zapisa i kontrolnog otiska. Ako se obrada prekine, ponovno
pokretanje ponavlja samo nedostajuće ili nevalidne serije. Galerijski izbori i razlozi
se čuvaju u `results/eda_review/*_gallery_selection.csv`.
