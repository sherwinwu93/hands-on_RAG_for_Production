# 脱敏
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# 方法检查出 敏感信息,并将其替换成typed placeholder. eg: <PERSON>,<PHONE_NUMBER>
def entity_aware_redaction(text):
    """检索敏感信息,并用占位符替代: eg: Sherwin -> <DOCTOR>"""
    analyzer = AnalyzerEngine()
    anoymizer = AnonymizerEngine()

    # 1. detect PII entities
    results = analyzer.analyze(text=text, entities=["PERSON", "PHONE_NUMBER"], language='en')

    # 2. Replace with Entity type(preserving semantic structure)
    anonymized_result = anoymizer.anonymize(
        text=text,
        analyzer_results= results,
        operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE_NUMBER>"})
        }
    )
    return anonymized_result.text

# test
input = "Dr. Bao called 123-555-1122."
output = entity_aware_redaction(input)
print(output)

