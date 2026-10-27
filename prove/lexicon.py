"""Static visual lexicon L_visual for ν (visual specificity).

Unigram anatomical, finding, and spatial terms in the spirit of RadLex / UMLS
(~300 terms). The lexicon is not learned and is not tuned per benchmark.

Paper Eq. 3 uses |tokens(q) ∩ L_visual|, i.e. set intersection of unigrams.
"""

from __future__ import annotations

VISUAL_LEXICON: frozenset[str] = frozenset(
    """
    abdomen abdominal acetabulum adrenal airway alveoli ankle aorta aortic
    appendix arm artery arteries articular atrium atria axilla axillary bladder
    bone bones bowel brachial brain brainstem breast breasts bronchus bronchi
    cardiac carina carotid cartilage cecum cerebellum cerebral cerebrum cervical
    cervix chest clavicle coccyx colon cortex cortical costal cranial cranium
    diaphragm disc disk duodenum dura elbow esophagus ethmoid extremity
    extremities facet fascia femur fibula finger fissure flank foot foramen fossa
    frontal gallbladder gastric globe groin hand head heart hepatic hilum hilar
    hip humerus ileum iliac ilium inguinal intestine intestines jaw jejunum joint
    jugular kidney kidneys knee larynx ligament liver lobe lobes lumbar lung lungs
    mandible maxilla mediastinum mediastinal meniscus meninges mitral mouth muscle
    muscles myocardium nasal nasopharynx neck nerve nerves nose occipital optic
    orbit orbital organ organs ovary ovaries palate pancreas pancreatic parietal
    parotid patella pelvis pelvic pericardium peritoneum pharynx pituitary pleura
    pleural pons portal prostate pubis pulmonary radius rectum renal retina rib
    ribs sacrum scapula shoulder sinus sinuses skull spine spleen splenic sternum
    stomach temporal tendon testis testes thalamus thoracic thorax thymus thyroid
    tibia tongue trachea ulna ureter urethra uterus valve vein veins ventricle
    ventricles vertebra vertebrae vertebral vessel vessels wrist
    abscess aneurysm atelectasis atrophy calcification calcifications calculus
    cardiomegaly cavity cirrhosis collapse collection congestion consolidation
    contusion cyst cysts deformity density deviation dilatation dilation
    displacement dissection distension edema effusion embolus emphysema
    enhancement erosion fibrosis finding findings fistula fluid fracture fractures
    hematoma hemorrhage hernia hydronephrosis hyperdensity hyperintensity
    hypodensity hypointensity infarct infarction infiltration inflammation
    ischemia lesion lesions lithiasis mass masses metastasis metastases nodule
    nodules obstruction opacity opacities osteophyte perforation pneumonia
    pneumothorax polyp protrusion rupture sclerosis stenosis stone stones
    stricture thickening thrombus tumor tumour ulcer abnormality abnormalities
    abnormal
    anterior anteroposterior apical apex axial bilateral caudal cephalad
    contralateral coronal distal dorsal hemisphere inferior ipsilateral lateral
    laterality left location located lower medial midline posterior proximal
    region right sagittal segment superior unilateral upper ventral where adjacent
    relative between above below within side basal central peripheral focal
    diffuse
    """.split()
)


def lexicon_size() -> int:
    return len(VISUAL_LEXICON)
