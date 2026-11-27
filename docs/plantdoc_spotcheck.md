# PlantDoc 类别映射抽查记录

- 抽查方式：每个 PlantDoc test 类别取字典序第一张图片，人工查看
- 抽查范围：test 27/27 类各 1 张（不是全量复核）；train 新增的 `Tomato two spotted spider mites leaf`（2 张）未抽查

## 结论

- **类别级映射合理**：test 27/27 类的映射均合理，新增的 spider mites 类按名称映射
- 发现的主要问题不是映射，而是 **PlantDoc 图像质量与类内标签噪声**：
  - 多张图片是教学幻灯片、网页截图或带水印图库照片，含文字/拼接图
  - 个别类别样例与“健康”目标不一致，或存在病害类型混淆
- 因此外部评估结果应结合图像质量与标签噪声解释；不建议把外部指标当作绝对性能

## 抽查明细

| PlantDoc 类别 | 样例文件 | 观察 | 映射置信度 | 备注 |
|---|---|---|---|---|
| Apple leaf | `20180511_090912-14gtw8a-e1526047952754.jpg` | 健康叶片 | 高 | 与 `Apple___healthy` 一致 |
| Apple rust leaf | `02.-Rust-2017-207u24s.jpg` | 锈斑 | 高 | 与 Cedar apple rust 一致 |
| Apple Scab Leaf | `052609%20Hartman%20Crabapple%20scab%20single%20leaf.JPG.jpg` | 疮痂病斑 | 高 | 一致 |
| Bell_pepper leaf | `10148582-green-leaf-of-pepper.jpg` | 健康叶片 | 高 | 与 healthy 一致 |
| Bell_pepper leaf spot | `00.jpg` | 严重叶斑/穿孔 | 高 | 与 Bacterial spot 一致 |
| Blueberry leaf | `blueberry-leaves-normal-above-and-iron-deficient-below-bgahf8.jpg` | 拼接图：正常叶 + 缺铁叶 | 中 | 映射到 healthy，但样例含缺铁表型 |
| Cherry leaf | `d7b79dbc0bca0eb3e63fbe5dfb256ccd53d90a4eb1413_1260x1260.jpg` | 健康叶片 | 高 | 与 healthy 一致 |
| Corn Gray leaf spot | `IMG_42231.jpg` | 灰斑病斑 | 高 | 与 Cercospora leaf spot 一致 |
| Corn leaf blight | `02c.jpg` | 长条状叶枯斑 | 高 | 与 Northern Leaf Blight 一致 |
| Corn rust leaf | `0796.39maizerust.jpg` | 锈斑 | 高 | 与 Common rust 一致 |
| grape leaf | `0000.jpg` | 健康叶片（白底） | 高 | 与 healthy 一致，图库风格 |
| grape leaf black rot | `03gb.jpg` | 果实 + 叶片病斑拼接 | 高 | 与 Black rot 一致 |
| Peach leaf | `00pe.jpg` | 健康叶片，但含文字标题 | 高（映射）/ 低（质量） | 网页图，文字可能干扰模型 |
| Potato leaf early blight | `3023.jpg` | 叶片枯斑；图内文字标注 *Phytophthora infestans* | 中 | 该文字对应晚疫病，**疑似标签噪声** |
| Potato leaf late blight | `1421_0.jpeg_itok=FMtmgePj.jpg` | 大片枯斑 | 高 | 与 Late blight 一致 |
| Raspberry leaf | `depositphotos_1323264-Raspberry-leaf-on-white.jpg` | 健康叶片，水印 | 高（映射）/ 低（质量） | 图库照片 |
| Soyabean leaf | `07feb_ma_sbr3.JPG.jpg` | 叶片有褐色斑点 | 中 | 映射到 healthy，但样例不似健康叶 |
| Squash Powdery mildew leaf | `powdery-mildew-erysiphe-plantani-on-young-sycamore-leaves-b774tm.jpg` | 白色粉状病斑 | 高 | 与 Powdery mildew 一致 |
| Strawberry leaf | `strawberry-leaf--stock-photo-1431216.jpg` | 健康叶片，水印 | 高（映射）/ 低（质量） | 图库照片 |
| Tomato Early blight leaf | `039b47d574bc4bb8a14259a1cd96a741.jpg` | 教学幻灯片（含文字/拼接） | 高（映射）/ 低（质量） | 非自然单叶图 |
| Tomato leaf | `1684.jpg` | 健康叶片 | 高 | 与 healthy 一致 |
| Tomato leaf bacterial spot | `%2320+Bacterial+Spot+and+Speck.jpg` | 教学幻灯片（含文字/果实图） | 高（映射）/ 低（质量） | 非自然单叶图 |
| Tomato leaf late blight | `110822-206-Tomato-blight.jpg` | 大片枯斑 | 高 | 与 Late blight 一致 |
| Tomato leaf mosaic virus | `9511.img.jpg` | 花叶/皱缩 | 高 | 与 mosaic virus 一致 |
| Tomato leaf yellow virus | `11-40580_5.jpg` | 黄化/皱缩 | 高 | 与 Yellow Leaf Curl Virus 一致 |
| Tomato mold leaf | `49%20Leafmold%20Bottom.jpg` | 叶面霉层 | 高 | 与 Leaf Mold 一致 |
| Tomato Septoria leaf spot | `0.jpg` | 教学幻灯片（含文字/拼接） | 高（映射）/ 低（质量） | 非自然单叶图 |

## 建议

1. **保留当前 27 类映射**，不因单张样例问题修改类别映射。
2. 在论文/PPT 中明确说明：PlantDoc 含教学幻灯片、网页截图和水印图库照片，外部评估存在额外域偏移。
3. 若需要更严格的外部评估，可后续人工筛选一个“干净子集”（排除文字/拼接/水印图），同时报告全量与干净子集结果。
4. 不建议只删除个别可疑图片后当作无偏评估；应在报告中说明筛选规则。
