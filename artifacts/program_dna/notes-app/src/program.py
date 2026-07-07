"""Clean-room scaffold generated from Program DNA evidence."""

TARGET_STACK = 'python'
FEATURE_CONFIDENCE = {
    'document_creation': 0.9299999999999999,
    'search_and_retrieval': 0.9299999999999999,
    'web_integration': 0.69,
    'automation': 0.61,
    'collaboration': 0.61,
    'legacy_migration': 0.53,
    'study_model': 0.69,
    'network_interaction': 0.53,
    'defensive_security_analysis': 0.61,
}

EVIDENCE_TO_CODE_TRACE = [
    {
        "candidate_modules": [
            "domain/document.py",
            "services/editor_controller.py",
            "adapters/ui_editor.py"
        ],
        "claim": "The replacement needs document_creation because the evidence shows this behavior or interface.",
        "code_obligations": [
            "Implement domain/document.py from clean-room behavior, not copied source.",
            "Implement services/editor_controller.py from clean-room behavior, not copied source.",
            "Implement adapters/ui_editor.py from clean-room behavior, not copied source."
        ],
        "evidence": [
            {
                "confidence": 0.72,
                "kind": "observed_behavior",
                "source": "observed_behavior:1",
                "summary": "Use Program DNA to reconstruct a notes app. Research open source alternatives, infer the architecture, build a scaffold workspace, include receipts, rollback, standards review, self-critique, sandbox/workspace boundaries, and compare it to engineering standards."
            },
            {
                "confidence": 0.7,
                "kind": "ui_affordance",
                "source": "ui_affordance:1",
                "summary": "Use Program DNA to reconstruct a notes app. Research open source alternatives, infer the architecture, build a scaffold workspace, include receipts, rollback, standards review, self-critique, sandbox/workspace boundaries, and compare it to engineering standards."
            },
            {
                "confidence": 0.76,
                "kind": "test_observation",
                "source": "test_observation:1",
                "summary": "Generate held-out behavior tests, UI workflow tests, golden-file tests, and failure-mode tests before claiming equivalence."
            },
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:1",
                "summary": "Research query `notes app architecture implementation language framework` returned 3 source/snippet record(s) via standard."
            },
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:2",
                "summary": "Research query `notes app open source alternative source code engineering` returned 3 source/snippet record(s) via standard."
            },
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:3",
                "summary": "Research query `how to build notes app app data model UI workflow` returned 3 source/snippet record(s) via standard."
            }
        ],
        "feature": "document_creation",
        "open_gap_policy": "If evidence is weak, emit a hypothesis and test; do not silently promote it to fact.",
        "research_obligations": [
            "Research comparable editor/document data models and undo/save semantics.",
            "Collect screenshots or UI traces for document lifecycle states."
        ],
        "test_obligations": [
            "Add held-out behavior tests proving document_creation works beyond examples.",
            "Add negative tests for malformed or unsupported document_creation inputs."
        ]
    },
    {
        "candidate_modules": [
            "services/search_index.py",
            "domain/query.py",
            "tests/search_equivalence.py"
        ],
        "claim": "The replacement needs search_and_retrieval because the evidence shows this behavior or interface.",
        "code_obligations": [
            "Implement services/search_index.py from clean-room behavior, not copied source.",
            "Implement domain/query.py from clean-room behavior, not copied source.",
            "Implement tests/search_equivalence.py from clean-room behavior, not copied source."
        ],
        "evidence": [
            {
                "confidence": 0.72,
                "kind": "observed_behavior",
                "source": "observed_behavior:1",
                "summary": "Use Program DNA to reconstruct a notes app. Research open source alternatives, infer the architecture, build a scaffold workspace, include receipts, rollback, standards review, self-critique, sandbox/workspace boundaries, and compare it to engineering standards."
            },
            {
                "confidence": 0.7,
                "kind": "ui_affordance",
                "source": "ui_affordance:1",
                "summary": "Use Program DNA to reconstruct a notes app. Research open source alternatives, infer the architecture, build a scaffold workspace, include receipts, rollback, standards review, self-critique, sandbox/workspace boundaries, and compare it to engineering standards."
            },
            {
                "confidence": 0.76,
                "kind": "test_observation",
                "source": "test_observation:1",
                "summary": "Generate held-out behavior tests, UI workflow tests, golden-file tests, and failure-mode tests before claiming equivalence."
            },
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:1",
                "summary": "Research query `notes app architecture implementation language framework` returned 3 source/snippet record(s) via standard."
            },
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:2",
                "summary": "Research query `notes app open source alternative source code engineering` returned 3 source/snippet record(s) via standard."
            },
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:3",
                "summary": "Research query `how to build notes app app data model UI workflow` returned 3 source/snippet record(s) via standard."
            }
        ],
        "feature": "search_and_retrieval",
        "open_gap_policy": "If evidence is weak, emit a hypothesis and test; do not silently promote it to fact.",
        "research_obligations": [
            "Research ranking/indexing algorithms appropriate to the data size and latency budget.",
            "Collect representative query/result examples."
        ],
        "test_obligations": [
            "Add held-out behavior tests proving search_and_retrieval works beyond examples.",
            "Add negative tests for malformed or unsupported search_and_retrieval inputs."
        ]
    },
    {
        "candidate_modules": [
            "adapters/http_client.py",
            "services/sync.py",
            "tests/web_contract.py"
        ],
        "claim": "The replacement needs web_integration because the evidence shows this behavior or interface.",
        "code_obligations": [
            "Implement adapters/http_client.py from clean-room behavior, not copied source.",
            "Implement services/sync.py from clean-room behavior, not copied source.",
            "Implement tests/web_contract.py from clean-room behavior, not copied source."
        ],
        "evidence": [
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:1",
                "summary": "Research query `notes app architecture implementation language framework` returned 3 source/snippet record(s) via standard."
            },
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:2",
                "summary": "Research query `notes app open source alternative source code engineering` returned 3 source/snippet record(s) via standard."
            },
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:3",
                "summary": "Research query `how to build notes app app data model UI workflow` returned 3 source/snippet record(s) via standard."
            }
        ],
        "feature": "web_integration",
        "open_gap_policy": "If evidence is weak, emit a hypothesis and test; do not silently promote it to fact.",
        "research_obligations": [
            "Research public API docs, auth flows, rate limits, offline behavior, and retry semantics.",
            "Capture request/response shapes without credentials or private payloads."
        ],
        "test_obligations": [
            "Add held-out behavior tests proving web_integration works beyond examples.",
            "Add negative tests for malformed or unsupported web_integration inputs."
        ]
    },
    {
        "candidate_modules": [
            "features/automation.py",
            "tests/test_automation.py"
        ],
        "claim": "The replacement needs automation because the evidence shows this behavior or interface.",
        "code_obligations": [
            "Implement features/automation.py from clean-room behavior, not copied source.",
            "Implement tests/test_automation.py from clean-room behavior, not copied source."
        ],
        "evidence": [
            {
                "confidence": 0.76,
                "kind": "test_observation",
                "source": "test_observation:1",
                "summary": "Generate held-out behavior tests, UI workflow tests, golden-file tests, and failure-mode tests before claiming equivalence."
            },
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:3",
                "summary": "Research query `how to build notes app app data model UI workflow` returned 3 source/snippet record(s) via standard."
            }
        ],
        "feature": "automation",
        "open_gap_policy": "If evidence is weak, emit a hypothesis and test; do not silently promote it to fact.",
        "research_obligations": [
            "Research implementation patterns and open-source analogs for automation.",
            "Collect held-out examples that would falsify a shallow automation implementation."
        ],
        "test_obligations": [
            "Add held-out behavior tests proving automation works beyond examples.",
            "Add negative tests for malformed or unsupported automation inputs."
        ]
    },
    {
        "candidate_modules": [
            "features/collaboration.py",
            "tests/test_collaboration.py"
        ],
        "claim": "The replacement needs collaboration because the evidence shows this behavior or interface.",
        "code_obligations": [
            "Implement features/collaboration.py from clean-room behavior, not copied source.",
            "Implement tests/test_collaboration.py from clean-room behavior, not copied source."
        ],
        "evidence": [
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:1",
                "summary": "Research query `notes app architecture implementation language framework` returned 3 source/snippet record(s) via standard."
            },
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:2",
                "summary": "Research query `notes app open source alternative source code engineering` returned 3 source/snippet record(s) via standard."
            }
        ],
        "feature": "collaboration",
        "open_gap_policy": "If evidence is weak, emit a hypothesis and test; do not silently promote it to fact.",
        "research_obligations": [
            "Research implementation patterns and open-source analogs for collaboration.",
            "Collect held-out examples that would falsify a shallow collaboration implementation."
        ],
        "test_obligations": [
            "Add held-out behavior tests proving collaboration works beyond examples.",
            "Add negative tests for malformed or unsupported collaboration inputs."
        ]
    },
    {
        "candidate_modules": [
            "features/legacy_migration.py",
            "tests/test_legacy_migration.py"
        ],
        "claim": "The replacement needs legacy_migration because the evidence shows this behavior or interface.",
        "code_obligations": [
            "Implement features/legacy_migration.py from clean-room behavior, not copied source.",
            "Implement tests/test_legacy_migration.py from clean-room behavior, not copied source."
        ],
        "evidence": [
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:3",
                "summary": "Research query `how to build notes app app data model UI workflow` returned 3 source/snippet record(s) via standard."
            }
        ],
        "feature": "legacy_migration",
        "open_gap_policy": "If evidence is weak, emit a hypothesis and test; do not silently promote it to fact.",
        "research_obligations": [
            "Research implementation patterns and open-source analogs for legacy_migration.",
            "Collect held-out examples that would falsify a shallow legacy_migration implementation."
        ],
        "test_obligations": [
            "Add held-out behavior tests proving legacy_migration works beyond examples.",
            "Add negative tests for malformed or unsupported legacy_migration inputs."
        ]
    },
    {
        "candidate_modules": [
            "features/study_model.py",
            "tests/test_study_model.py"
        ],
        "claim": "The replacement needs study_model because the evidence shows this behavior or interface.",
        "code_obligations": [
            "Implement features/study_model.py from clean-room behavior, not copied source.",
            "Implement tests/test_study_model.py from clean-room behavior, not copied source."
        ],
        "evidence": [
            {
                "confidence": 0.72,
                "kind": "observed_behavior",
                "source": "observed_behavior:1",
                "summary": "Use Program DNA to reconstruct a notes app. Research open source alternatives, infer the architecture, build a scaffold workspace, include receipts, rollback, standards review, self-critique, sandbox/workspace boundaries, and compare it to engineering standards."
            },
            {
                "confidence": 0.7,
                "kind": "ui_affordance",
                "source": "ui_affordance:1",
                "summary": "Use Program DNA to reconstruct a notes app. Research open source alternatives, infer the architecture, build a scaffold workspace, include receipts, rollback, standards review, self-critique, sandbox/workspace boundaries, and compare it to engineering standards."
            },
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:1",
                "summary": "Research query `notes app architecture implementation language framework` returned 3 source/snippet record(s) via standard."
            }
        ],
        "feature": "study_model",
        "open_gap_policy": "If evidence is weak, emit a hypothesis and test; do not silently promote it to fact.",
        "research_obligations": [
            "Research implementation patterns and open-source analogs for study_model.",
            "Collect held-out examples that would falsify a shallow study_model implementation."
        ],
        "test_obligations": [
            "Add held-out behavior tests proving study_model works beyond examples.",
            "Add negative tests for malformed or unsupported study_model inputs."
        ]
    },
    {
        "candidate_modules": [
            "adapters/network_monitor.py",
            "tests/network_boundary.py"
        ],
        "claim": "The replacement needs network_interaction because the evidence shows this behavior or interface.",
        "code_obligations": [
            "Implement adapters/network_monitor.py from clean-room behavior, not copied source.",
            "Implement tests/network_boundary.py from clean-room behavior, not copied source."
        ],
        "evidence": [
            {
                "confidence": 0.66,
                "kind": "research_result",
                "source": "program_dna:research:3",
                "summary": "Research query `how to build notes app app data model UI workflow` returned 3 source/snippet record(s) via standard."
            }
        ],
        "feature": "network_interaction",
        "open_gap_policy": "If evidence is weak, emit a hypothesis and test; do not silently promote it to fact.",
        "research_obligations": [
            "Research implementation patterns and open-source analogs for network_interaction.",
            "Collect held-out examples that would falsify a shallow network_interaction implementation."
        ],
        "test_obligations": [
            "Add held-out behavior tests proving network_interaction works beyond examples.",
            "Add negative tests for malformed or unsupported network_interaction inputs."
        ]
    },
    {
        "candidate_modules": [
            "security/threat_model.py",
            "security/forensics.py",
            "tests/security_boundary.py"
        ],
        "claim": "The replacement needs defensive_security_analysis because the evidence shows this behavior or interface.",
        "code_obligations": [
            "Implement security/threat_model.py from clean-room behavior, not copied source.",
            "Implement security/forensics.py from clean-room behavior, not copied source.",
            "Implement tests/security_boundary.py from clean-room behavior, not copied source."
        ],
        "evidence": [
            {
                "confidence": 0.72,
                "kind": "observed_behavior",
                "source": "observed_behavior:1",
                "summary": "Use Program DNA to reconstruct a notes app. Research open source alternatives, infer the architecture, build a scaffold workspace, include receipts, rollback, standards review, self-critique, sandbox/workspace boundaries, and compare it to engineering standards."
            },
            {
                "confidence": 0.7,
                "kind": "ui_affordance",
                "source": "ui_affordance:1",
                "summary": "Use Program DNA to reconstruct a notes app. Research open source alternatives, infer the architecture, build a scaffold workspace, include receipts, rollback, standards review, self-critique, sandbox/workspace boundaries, and compare it to engineering standards."
            }
        ],
        "feature": "defensive_security_analysis",
        "open_gap_policy": "If evidence is weak, emit a hypothesis and test; do not silently promote it to fact.",
        "research_obligations": [
            "Research observable indicators of compromise and relevant defensive signatures.",
            "Keep payload reproduction non-deployable and forensic-only."
        ],
        "test_obligations": [
            "Add held-out behavior tests proving defensive_security_analysis works beyond examples.",
            "Add negative tests for malformed or unsupported defensive_security_analysis inputs."
        ]
    }
]

class ReconstructedProgram:
    def __init__(self):
        self.receipts = []

    def capabilities(self):
        return sorted(FEATURE_CONFIDENCE)

    def evidence_trace(self, feature=None):
        if feature is None:
            return list(EVIDENCE_TO_CODE_TRACE)
        return [item for item in EVIDENCE_TO_CODE_TRACE if item.get('feature') == feature]

    def execute(self, feature, payload=None):
        if feature not in FEATURE_CONFIDENCE:
            raise ValueError(f'unknown reconstructed feature: {feature}')
        receipt = {
            'feature': feature,
            'payload': payload or {},
            'status': 'planned',
            'evidence_trace': self.evidence_trace(feature),
        }
        self.receipts.append(receipt)
        return receipt
