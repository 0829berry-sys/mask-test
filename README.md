# 街景盆栽批次遮罩 / 選擇性上色工具

單機 Streamlit 應用，用於碩論視覺刺激物製作：辨識街景照片中的盆栽（植物＋花盆／支架），
保留其原始色彩，其餘背景轉為亮度加權灰階，並支援互動式遮罩精修與批次匯出。

---

## 1. 架構選擇說明

| | Python + Streamlit（本工具採用） | 純瀏覽器 React/Vue + WebAssembly |
|---|---|---|
| 高解析度大圖處理 | ✅ 直接用 NumPy/OpenCV 處理原始陣列，記憶體與運算不受瀏覽器分頁限制；4000×3000px 以上影像也可穩定處理 | ⚠️ 受限於瀏覽器分頁記憶體與單執行緒 JS（WASM 多執行緒需額外設定），大圖容易卡頓或崩潰 |
| 分割模型精準度 | ✅ 可直接跑原生 PyTorch 版 SAM（`vit_h`/`vit_l`/`vit_b`），精準度最佳；有 GPU 可大幅加速 | ⚠️ 需轉為 ONNX 並用 onnxruntime-web 執行，通常僅能跑輕量版（MobileSAM），精準度打折 |
| 標註流暢度 | ⚠️ Streamlit 每次互動會重跑整個 script，畫筆連續拖曳的即時回饋不如原生 Canvas app 絲滑（已用縮小顯示圖＋批次套用筆畫緩解） | ✅ 原生 Canvas + requestAnimationFrame，畫筆體驗最流暢 |
| 批次檔案與磁碟存取 | ✅ 本機執行、直接讀寫檔案系統，適合大量圖檔與研究資料夾管理 | ⚠️ 瀏覽器沙盒限制，需靠 File System Access API（僅部分瀏覽器支援）或逐檔下載 |
| 開發/維護成本 | ✅ 單一語言（Python），科研人員最熟悉的技術棧，模組化易擴充 | ⚠️ 需前後端分工或熟悉 WASM 打包流程，維護成本較高 |

**結論**：由於研究刺激物對「解析度保留」與「分割精準度」要求極為嚴謹，且你先前已在
Streamlit 環境下工作過，這裡採用 **Python + Streamlit + 原生 SAM** 作為主線，並內建
OpenCV GrabCut 作為免下載模型的輕量備援引擎（適合快速產出草稿或沒有 GPU 的情境）。
若日後需要多人協作透過瀏覽器共用標註（而非單機使用），屆時再評估遷移至 Web 架構會更合理。

---

## 2. 安裝

```bash
# 建議使用虛擬環境
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2.1 下載 SAM 模型權重（若使用 SAM 引擎）

至 Meta 官方 Segment Anything repo 下載 checkpoint（三種大小任選一種，`vit_b` 最快最小，建議先用這個）：

- `vit_b`：`sam_vit_b_01ec64.pth`（約 375 MB）
- `vit_l`：`sam_vit_l_0b3195.pth`（約 1.2 GB）
- `vit_h`：`sam_vit_h_4b8939.pth`（約 2.4 GB，最精準但最慢）

下載連結請於 GitHub 搜尋 `facebookresearch/segment-anything`，於 README 的 Model Checkpoints
區塊取得（連結會隨版本更新，故不在此寫死網址）。下載後放到專案內的 `checkpoints/` 資料夾，
或在側邊欄輸入你自己的存放路徑。

若你的 Windows 桌機有 NVIDIA GPU，安裝對應 CUDA 版 PyTorch 可大幅加速（到
[pytorch.org](https://pytorch.org/get-started/locally/) 依你的 CUDA 版本取得安裝指令，取代
requirements.txt 中的 CPU 版 `torch`）；側邊欄「運算裝置」選 `cuda` 即可。

**不想安裝 SAM/torch？** 側邊欄分割引擎切換為「GrabCut」即可完全跳過本節，
GrabCut 已內建於 `opencv-python-headless`，免額外下載模型。

### 2.2 HEIC/HEIF 支援

若不需要讀取 iPhone 的 `.heic` 檔，可從 `requirements.txt` 移除 `pillow-heif`，其餘功能不受影響。

---

## 3. 執行

```bash
streamlit run app.py
```

瀏覽器會自動開啟（預設 `http://localhost:8501`）。此為本機伺服器，資料不會上傳到外部。

---

## 4. 使用流程

1. **上傳影像**：左側側邊欄拖曳或選取多張 JPG/PNG/TIFF（HEIC 需安裝 pillow-heif），
   縮圖佇列會顯示每張的狀態（未處理／待確認／已校正完成）。
2. **選擇引擎**：側邊欄選 SAM（需先設定 checkpoint 路徑）或 GrabCut。
3. **點選提示產生初始遮罩**：工具選「SAM／GrabCut 點選提示」，在盆栽上點藍色正樣本、
   在背景誤入區域點紅色負樣本，畫面下方按「✅ 加入目前物件至遮罩」確認。可重複此流程
   標註同一張圖中的多個盆栽。
4. **手動精修**：切換工具為「畫筆（新增）」補齊遺漏葉片，或「橡皮擦（移除）」剔除誤標，
   調整筆刷大小後在畫布上塗抹，完成一段筆畫後按「套用目前筆畫」。
5. **檢查**：右上「預覽模式」可切換「原圖＋遮罩疊圖」「最終選擇性上色預覽」「純黑白二值遮罩」，
   並用「邊緣羽化半徑」滑桿調整邊界柔化程度（建議 1–3px）。
6. **Undo/Redo**：每次確認遮罩或套用筆畫前都會存檔，可用對應按鈕復原／重做。
7. **完成並下一張**：標記目前影像為「已校正完成」並跳至下一張。
8. **批次匯出**：底部「批次匯出」區塊選擇範圍（僅完成／全部）、是否同步輸出二值遮罩，
   可直接存至本機資料夾（因為此工具就在你的電腦上執行），或產生 ZIP 直接下載，
   兩者都會附上 `processing_log.json`，記錄每張圖的遮罩像素佔比與處理時間戳記。

---

## 5. 已知限制與可擴充方向

- **資料夾選取**：瀏覽器安全限制下，Streamlit 的檔案上傳僅能「多選檔案」，無法遞迴選取整個
  資料夾樹狀結構；實務上在系統檔案對話框中全選資料夾內檔案即可達到相近效果。
- **鍵盤快捷鍵**：目前以按鈕取代（Undo/Redo/上一張/下一張/套用筆畫），未綁定
  Ctrl+Z / [ ] / Enter 等鍵盤事件——Streamlit 原生不支援全域鍵盤監聽，如需要可另外整合
  `streamlit-shortcuts` 套件或自訂 JS 元件，但穩定性因版本而異，故先以按鈕介面確保可用性。
- **筆畫即時感**：Streamlit 每次互動會重跑整個 script，因此畫筆採「畫完一段、按套用」的
  批次確認模式，而非逐像素即時更新遮罩；如需更絲滑的畫筆體驗，可考慮之後把「手動精修」
  這一步驟獨立遷出成一個小型 React canvas 元件。
- **多物件追蹤**：目前遮罩為單一二值圖層（盆栽 vs. 背景），未個別標記「第幾個盆栽」；
  若論文需要逐一計算每個盆栽的面積占比，可在 `export_one()` 中改用
  `cv2.connectedComponents` 對最終遮罩做實例編號後再輸出。
