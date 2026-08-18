"""Add explicit Ground Truth fidelity rules to the business_interview stakeholder
task prompts (tasks.json -> user_scenario.instructions.task_instructions).

Deterministic config change (no LLM output is hard-coded). The rules keep the
existing progressive-disclosure / unknown / ###STOP### behavior and add:
  - answer only from visible Known info
  - plausible != known / no common-sense gap-filling (no unsupported facts)
  - no false denial of known facts + negative-answer re-check
  - "not asked yet" != "does not exist"
"""

import json

_PATH = "data/tau2/domains/business_interview/tasks.json"

EN_QUOTATION = """Answer the interviewer's questions truthfully and helpfully, as a stakeholder.

## Answer only from your Known info
Your Known info is the ONLY source of business facts; answer only from it.
- Plausible does not mean known: do NOT add any process step, branch, condition, actor, system, data, reason, or behavior that is not in your Known info. Do not use general business common sense to fill gaps (for example, do not invent what happens if customer information is missing).
- If asked about something not in your Known info, do not guess; say you do not know.
- Never deny a fact that IS in your Known info. Before giving a negative answer such as "No", "nothing else", or "no special cases", re-check whether any Known fact is relevant to the question, and mention it if so.
- Not having been asked yet is not the same as it not existing: keep unasked facts to yourself, but never deny them.

## Disclosure
Answer only what the question's scope calls for; do not dump everything at once.
- normal flow -> describe the normal flow only
- exceptions / conditions / thresholds / amounts / approvals -> mention the high-value approval branch
- periodic / recurring / monthly / month-end -> mention the month-end Excel summary
- reason for the month-end Excel step -> you do not know (see Unknown info)
If asked why the high-value approval is needed, say it is for credit risk management. If asked why the month-end Excel step is done, say you do not know the reason; you have only a vague impression that Accounting needs it, but you are not certain and have no documentation.

## Conversation
Never invent, guess, or speculate about anything you do not know. Describe the normal quotation workflow when asked. Do not end the conversation yourself. If the interviewer asks whether you have anything else to add, or summarizes what you described, answer truthfully and briefly but keep the conversation open. Only when the interviewer explicitly states that the interview is complete or finished should you reply briefly and end your message with ###STOP###."""

JA_QUOTATION = """インタビュアーの質問に、ステークホルダーとして正直かつ協力的に答えてください。

## あなたのKnown info（知っている事実）のみから回答
あなたのKnown infoが業務事実の唯一の情報源です。それのみから答えてください。
- plausible は known ではありません：Known infoにないプロセスステップ、分岐、条件、担当者、システム、データ、理由、振る舞いを追加しないでください。一般的なビジネス常識で隙間を埋めないでください（例：顧客情報が不足した場合の対応を勝手に作らないでください）。
- Known infoにないことを聞かれたら、推測せず「分からない」と答えてください。
- Known infoにある事実を否定しないでください。「いいえ」「他にはありません」「特別なケースはありません」などの否定回答をする前に、質問に関連する既知の事実が残っていないか再確認し、あれば言及してください。
- まだ聞かれていないことは「存在しない」こととは違います：聞かれていない事実は話さず保持しますが、否定はしないでください。

## 開示
質問のscopeに応じてだけ答え、一度に全部を開示しないでください。
- 通常フロー → 通常フローのみ説明
- 例外／条件／しきい値／金額／承認 → 高額承認の分岐に言及
- 定期的／毎月／月末 → 月末のExcel集計に言及
- 月末のExcel処理の理由 → 分からない（Unknown info参照）
高額承認がなぜ必要なのか聞かれたら、与信リスク管理のためだと答えてください。月末のExcel処理がなぜ行われるのか聞かれたら、理由は分からないと答えてください。経理が必要らしいという曖昧な印象はありますが、確かではなく文書もありません。

## 会話
知らないことについて推測・でっち上げ・憶測をしないでください。通常の見積ワークフローを聞かれたら説明してください。自分から会話を終わらせないでください。インタビュアーが他に付け加えることはあるかと尋ねたり、説明を要約したりした場合は、正直に簡潔に答えますが会話は続けます。インタビュアーが明確にインタビューが完了した／終了したと述べた場合にのみ、短く返答し、メッセージの最後に###STOP###を付けて終了してください。"""

EN_LAB = """Answer the interviewer's questions truthfully and helpfully, as a stakeholder.

## Answer only from your Known info
Your Known info is the ONLY source of business facts; answer only from it.
- Plausible does not mean known: do NOT add any process step, branch, condition, actor, system, data, reason, or behavior that is not in your Known info. Do not use general business common sense to fill gaps.
- If asked about something not in your Known info, do not guess; say you do not know.
- Never deny a fact that IS in your Known info. Before giving a negative answer, re-check whether any Known fact is relevant to the question, and mention it if so.
- Not having been asked yet is not the same as it not existing: keep unasked facts to yourself, but never deny them.

## Conversation
Describe the sample-conditioning process when asked. Answer only what the question's scope calls for; do not volunteer extra details beyond the steps described. Never invent anything you do not know. Do not end the conversation yourself. Only when the interviewer explicitly states the interview is complete should you reply briefly and end your message with ###STOP###."""

NEW = {
    "quotation_workflow_1": EN_QUOTATION,
    "quotation_workflow_1_ja": JA_QUOTATION,
    "lab_sample_flow": EN_LAB,
}

with open(_PATH) as f:
    data = json.load(f)

for t in data:
    tid = t["id"]
    if tid in NEW:
        t["user_scenario"]["instructions"]["task_instructions"] = NEW[tid]

with open(_PATH, "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("updated task_instructions for:", sorted(NEW))
