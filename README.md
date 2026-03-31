# CO3133 - Deep Learning and Its Applications

Workspace nay duoc khoi tao de lam cac bai tap lon theo de trong file `assignment-vne.pdf`.

## 1) Muc tieu BTL 1

Thuc hien bai toan classification tren 3 loai du lieu:

1. Image classification
2. Text classification
3. Multimodal classification (image + text)

## 2) Rang buoc bat buoc theo de

- Moi dataset co it nhat 5 lop.
- Dataset du lon de danh gia thuyet phuc (uu tien tu vai nghin mau tro len).
- Multimodal dataset phai la cap image-text dung nghia (khong ghep ngau nhien).

## 3) So sanh ky thuat can co

- Image: CNN vs ViT (pretrained + fine-tuning).
- Text: RNN/LSTM vs Transformer.
- Multimodal: Zero-shot vs Few-shot.

Metric toi thieu: Accuracy. Neu mat can bang lop can bo sung F1-score.

## 4) Cau truc du an

```
.
|-- assignment-vne.pdf
|-- README.md
|-- requirements.txt
|-- docs/
|   `-- CHECKLIST.md
|-- index.html
|-- style.css
|-- main.js
|-- assignment1.html
|-- assignment2.html
|-- assignment3.html
|-- btl_mr.html
|-- assets/
|   `-- .gitkeep
|-- assignment1/
|   `-- README.md
|-- assignment2/
|   `-- README.md
|-- assignment3/
|   `-- README.md
|-- btl_mr/
|   `-- README.md
`-- notebooks/
		|-- BTL1_deep_learning.ipynb
		|-- BTL1_traditional.ipynb
		|-- BTL2_deep_learning.ipynb
		|-- BTL2_traditional.ipynb
		|-- BTL3_deep_learning.ipynb
		|-- BTL3_traditional.ipynb
		`-- BTL_Extra_HMM.ipynb
```

## 5) Landing Page (GitHub Pages)

Trang chu va cac trang assignment da duoc tao khung:

- `index.html`: thong tin nhom, thanh vien, giang vien, link den cac bai tap.
- `assignment1.html`, `assignment2.html`, `assignment3.html`, `btl_mr.html`:
	trang rieng cho tung bai.

Can cap nhat them:

- Link video demo.
- Link video trinh bay (YouTube).
- Link code.
- Link bao cao EDA, dataset/dataloader/augmentation, huan luyen va so sanh.

## 6) Tien do de xuat

1. Chot 3 dataset dap ung rang buoc.
2. Viet notebook baseline cho tung bai toan.
3. Fine-tune va danh gia theo cap mo hinh bat buoc.
4. Tong hop bang so lieu, bieu do, phan tich loi.
5. Bo sung mot huong mo rong de tang diem.
6. Cap nhat landing page va video.

## 7) Moc han theo de

- Bao cao lan 1: 23:59 - 26/03/2026.
- Bao cao cuoi: 23:59 - 06/04/2026.
- Tre han: tru 20% diem phan trinh bay moi tuan.

## 8) Cai dat nhanh

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## 9) Ghi chu

- Dung `docs/CHECKLIST.md` de bam sat de va kiem tra truoc khi nop.
- Khong commit du lieu qua lon truc tiep vao git repo.
