# 🔄 Sim2Real Image Translation – Documentation

---

## 1. مقدمة المشروع

يهدف هذا المشروع إلى تحويل الصور الناتجة عن المحاكاة (Simulation Data) إلى صور واقعية (Realistic Data) باستخدام تقنيات الذكاء الاصطناعي الحديثة، خاصة في المجالات المرتبطة بـ:

- الرؤية الحاسوبية (Computer Vision)
- الاستشعار عن بعد (Remote Sensing)
- الأقمار الصناعية والتحليل الطيفي
- التطبيقات الصناعية التي تعتمد على بيانات اصطناعية لتحسين أداء النماذج

هذا المشروع ينتمي إلى مجال **Sim2Real Domain Adaptation**، حيث نعمل على تقليل الفجوة بين بيانات المحاكاة والبيانات الواقعية، مما يسمح للنماذج المدربة على بيانات المحاكاة أن تعمل بشكل أفضل على بيانات العالم الحقيقي.

---

## 2. الأدوات والتقنيات المستخدمة

| التقنية                  | الوصف                                                      | ملاحظات                            |
|--------------------------|------------------------------------------------------------|-----------------------------------|
| Neural Style Transfer (NST) | نقل النمط البصري من صورة حقيقية إلى صورة محاكاة، مع الحفاظ على الهيكل | لا يحتاج صور متقابلة، لتحسين الشكل فقط |
| CycleGAN                 | تحويل صور بين مجاليْن بدون الحاجة لبيانات متقابلة (Unpaired) | مناسب لبيانات الاستشعار عن بعد والصور التي يصعب الحصول على زوج لها |
| Pix2Pix                  | تحويل صور معتمدة على بيانات متقابلة (Paired Image-to-Image) | دقة عالية مع توفر بيانات متقابلة  |
| Python + PyTorch         | بيئة برمجية لتنفيذ وتدريب النماذج                           | دعم مجتمعي قوي وأدوات جاهزة       |
| Google Colab             | منصة سحابية لتجربة النماذج بدون الحاجة لموارد محلية         | مناسب لتجربة النماذج في بيئة مجانية |

---

## 3. شرح التقنيات

### 3.1 Neural Style Transfer (NST)

- **الهدف:** نقل النمط البصري (مثل الألوان، والملمس) من صورة واقعية إلى صورة محاكاة مع المحافظة على المحتوى الأصلي.
- **المميزات:**
  - لا يحتاج صور متقابلة (Unpaired).
  - يحسن المظهر البصري فقط، لا يقوم بتحويل شامل بين المجالات.
- **العيوب:**
  - لا يعالج الاختلافات الهيكلية أو التوزيعية بين المجالين.
- **روابط مهمة:**
  - [PyTorch NST Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)

---

### 3.2 CycleGAN

- **الهدف:** ترجمة الصور بين مجالين بدون الحاجة إلى صور متقابلة (Unpaired Image-to-Image Translation).
- **كيف يعمل؟**
  - يتضمن شبكتين مولدتين (`Generator G: A→B` و `Generator F: B→A`) لتوليد صور من كل مجال إلى الآخر.
  - يستخدم أيضًا شبكتين مميزتين (`Discriminator D_A` و `Discriminator D_B`) لتمييز الصور الحقيقية من الصور المولدة.
  - **Cycle Consistency Loss** يضمن أنه عند تحويل صورة من A إلى B ثم العودة إلى A، نحصل على صورة مشابهة للأصل.
- **المزايا:**
  - لا يحتاج لبيانات تدريب متقابلة.
  - مناسب جدًا للاستشعار عن بعد والبيانات التي يصعب إيجاد صور paired لها.
- **المصادر:**
  - الورقة البحثية الأصلية: [CycleGAN: Unpaired Image-to-Image Translation](https://arxiv.org/pdf/1703.10593)
  - الموقع الرسمي: [https://junyanz.github.io/CycleGAN/](https://junyanz.github.io/CycleGAN/)
  - GitHub: [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
  - شرح تفصيلي: [Viso.ai CycleGAN](https://viso.ai/deep-learning/cyclegan/)
  - شرح ArcGIS: [How CycleGAN Works](https://developers.arcgis.com/python/latest/guide/how-cyclegan-works/)

---

### 3.3 Pix2Pix

- **الهدف:** تحويل صور بين مجالين باستخدام بيانات متقابلة (Paired Image-to-Image Translation).
- **كيف يعمل؟**
  - يستخدم شبكة مولدة (Generator) تقوم بتحويل الصورة من المجال المصدر إلى الهدف.
  - شبكة مميزة (Discriminator) تميز بين الصور الحقيقية والمولدة.
  - يستخدم خسارة L1 للحفاظ على التشابه مع الصورة الأصلية.
- **المزايا:**
  - دقة عالية في التحويل عند توفر بيانات paired.
- **العيوب:**
  - يتطلب وجود بيانات متقابلة لكل زوج صورة simulation مع صورتها الواقعية.
- **المصدر:**
  - GitHub: [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

---

## 4. هيكلية المشروع (Structure)

```bash
project/
├── datasets/
│   └── your_dataset/
│       ├── trainA/       # صور simulation (مجال A)
│       ├── trainB/       # صور real (مجال B)
│       ├── testA/        # صور simulation للاختبار
│       └── testB/        # صور real للاختبار
├── models/               # تخزين النماذج المدربة
├── results/              # نتائج التحويل والتقييم
├── notebooks/            # دفاتر Jupyter / Colab
│   ├── NST.ipynb
│   ├── CycleGAN_Train.ipynb
│   └── Pix2Pix_Train.ipynb
├── train.py              # سكريبت تدريب
├── test.py               # سكريبت اختبار
└── README.md             # ملف التوثيق
# Sim_2_Real