import fitz  # PyMuPDF

pdf_path = "/Users/kasunachinthaperera/Documents/Final Year Project:Thesis/NEXUS Brain VHT/figures/NEXUS Brain VHT - Simulation & MRI Dashboard 22.58.49.pdf"
doc = fitz.open(pdf_path)

for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=300)
    output_filename = f"nexus_dashboard_page_{i+1}.png"
    pix.save(output_filename)
    print(f"Saved: {output_filename}")