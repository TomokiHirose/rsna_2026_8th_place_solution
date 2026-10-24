# 12 個の二値 finding。train.csv と sample_submission.csv が使う列順そのまま。
# 自前で並べ直さず、このリストを import すること。
LABELS: list[str] = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

NUM_LABELS: int = len(LABELS)

# 提出 / test の識別子の列。
ID_COLUMN: str = "StudyInstanceUID"
