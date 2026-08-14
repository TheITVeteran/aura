# Aura Architecture Dependency Map

*Reviewed against the tree: 2026-08-13. See [documentation status map](DOC_STATUS.md) for how to read this file.*

Schema: `aura.architecture.dependency_map.v2`
Root: `<AURA_ROOT>`
Generated: `0.0`

## Summary

- Subsystems: 153
- Python files: 2741
- Python lines: 1190736
- Dependency edges: 1198
- ServiceContainer `.get()` calls: 1463
- ServiceContainer registrations: 337
- Boot contract: PASS

## Subsystem Dependency Graph

```mermaid
graph TD
    runtime["runtime<br/>213 files, 88663 lines"]
    utils["utils<br/>44 files, 7557 lines"]
    brain["brain<br/>350 files, 250682 lines"]
    memory["memory<br/>104 files, 31718 lines"]
    consciousness["consciousness<br/>155 files, 75756 lines"]
    resilience["resilience<br/>62 files, 17934 lines"]
    governance["governance<br/>14 files, 6633 lines"]
    security["security<br/>51 files, 14522 lines"]
    health["health<br/>7 files, 2043 lines"]
    conversation["conversation<br/>42 files, 25642 lines"]
    observability["observability<br/>14 files, 4541 lines"]
    agency["agency<br/>50 files, 21892 lines"]
    executive["executive<br/>15 files, 6834 lines"]
    perception["perception<br/>40 files, 17455 lines"]
    senses["senses<br/>31 files, 9214 lines"]
    affect["affect<br/>12 files, 4731 lines"]
    epistemics["epistemics<br/>18 files, 6615 lines"]
    identity["identity<br/>19 files, 3755 lines"]
    adaptation["adaptation<br/>28 files, 16270 lines"]
    cognition["cognition<br/>26 files, 10614 lines"]
    self_modification["self_modification<br/>35 files, 13396 lines"]
    state["state<br/>7 files, 4791 lines"]
    knowledge["knowledge<br/>14 files, 4103 lines"]
    skills["skills<br/>81 files, 34633 lines"]
    being["being<br/>25 files, 7521 lines"]
    learning["learning<br/>163 files, 109770 lines"]
    verify["verify<br/>17 files, 6265 lines"]
    world_model["world_model<br/>11 files, 3778 lines"]
    autonomy["autonomy<br/>29 files, 14280 lines"]
    orchestrator["orchestrator<br/>42 files, 22757 lines"]
    organism["organism<br/>9 files, 4763 lines"]
    social["social<br/>20 files, 8247 lines"]
    voice["voice<br/>36 files, 13847 lines"]
    continuity["continuity<br/>1 files, 33 lines"]
    values["values<br/>15 files, 2062 lines"]
    bus["bus<br/>7 files, 4191 lines"]
    capabilities["capabilities<br/>20 files, 15243 lines"]
    goals["goals<br/>12 files, 4424 lines"]
    reasoning["reasoning<br/>16 files, 7752 lines"]
    morality["morality<br/>16 files, 1327 lines"]
    sandbox["sandbox<br/>5 files, 1328 lines"]
    self["self<br/>10 files, 5690 lines"]
    tasks["tasks<br/>4 files, 566 lines"]
    embodiment["embodiment<br/>28 files, 13531 lines"]
    introspection["introspection<br/>10 files, 3330 lines"]
    phases["phases<br/>29 files, 23086 lines"]
    actuators["actuators<br/>11 files, 5142 lines"]
    autonomic["autonomic<br/>6 files, 3714 lines"]
    discovery["discovery<br/>7 files, 2151 lines"]
    fsw["fsw<br/>7 files, 2968 lines"]
    kernel["kernel<br/>11 files, 6779 lines"]
    ops["ops<br/>17 files, 5496 lines"]
    planning["planning<br/>9 files, 4408 lines"]
    world["world<br/>24 files, 1489 lines"]
    cognitive["cognitive<br/>12 files, 9736 lines"]
    coordinators["coordinators<br/>10 files, 4959 lines"]
    ethics["ethics<br/>2 files, 602 lines"]
    llm["llm<br/>3 files, 259 lines"]
    managers["managers<br/>6 files, 957 lines"]
    ontogeny["ontogeny<br/>17 files, 7037 lines"]
    pipeline["pipeline<br/>3 files, 808 lines"]
    resource["resource<br/>4 files, 624 lines"]
    sleep["sleep<br/>10 files, 1260 lines"]
    somatic["somatic<br/>6 files, 2944 lines"]
    supervisor["supervisor<br/>3 files, 993 lines"]
    unity["unity<br/>11 files, 2825 lines"]
    advanced_cognition["advanced_cognition<br/>13 files, 5381 lines"]
    agi["agi<br/>6 files, 2847 lines"]
    collective["collective<br/>6 files, 2204 lines"]
    data["data<br/>3 files, 652 lines"]
    dialogue["dialogue<br/>4 files, 745 lines"]
    environment["environment<br/>76 files, 9153 lines"]
    evaluation["evaluation<br/>19 files, 4449 lines"]
    media["media<br/>4 files, 964 lines"]
    meta["meta<br/>7 files, 1278 lines"]
    motivation["motivation<br/>7 files, 1209 lines"]
    promotion["promotion<br/>6 files, 936 lines"]
    reality_reach["reality_reach<br/>33 files, 30903 lines"]
    self_improvement["self_improvement<br/>21 files, 9144 lines"]
    soma["soma<br/>4 files, 1532 lines"]
    adapters["adapters<br/>5 files, 2265 lines"]
    conversational["conversational<br/>4 files, 2435 lines"]
    db["db<br/>4 files, 600 lines"]
    ghost["ghost<br/>6 files, 2076 lines"]
    morphogenesis["morphogenesis<br/>11 files, 3112 lines"]
    phenomenal_substrate["phenomenal_substrate<br/>11 files, 1042 lines"]
    pneuma["pneuma<br/>7 files, 1279 lines"]
    search["search<br/>2 files, 1939 lines"]
    verification["verification<br/>4 files, 350 lines"]
    workspace["workspace<br/>9 files, 1243 lines"]
    architect["architect<br/>25 files, 7121 lines"]
    coherence["coherence<br/>2 files, 407 lines"]
    evals["evals<br/>2 files, 686 lines"]
    evolution["evolution<br/>8 files, 2131 lines"]
    grounding["grounding<br/>8 files, 1263 lines"]
    lattice["lattice<br/>5 files, 704 lines"]
    maintenance["maintenance<br/>2 files, 295 lines"]
    persistence["persistence<br/>2 files, 619 lines"]
    plasticity["plasticity<br/>5 files, 428 lines"]
    predictive["predictive<br/>2 files, 186 lines"]
    sensors["sensors<br/>1 files, 195 lines"]
    services["services<br/>2 files, 31 lines"]
    sim["sim<br/>2 files, 452 lines"]
    simulation["simulation<br/>3 files, 402 lines"]
    sovereign["sovereign<br/>4 files, 522 lines"]
    startup["startup<br/>4 files, 546 lines"]
    unknowns["unknowns<br/>4 files, 325 lines"]
    actuation["actuation<br/>9 files, 1130 lines"]
    architecture_quality["architecture_quality<br/>6 files, 1979 lines"]
    audit["audit<br/>7 files, 1016 lines"]
    body["body<br/>22 files, 1778 lines"]
    communication["communication<br/>5 files, 2193 lines"]
    consent["consent<br/>2 files, 167 lines"]
    context["context<br/>6 files, 1984 lines"]
    creativity["creativity<br/>2 files, 801 lines"]
    curriculum["curriculum<br/>7 files, 658 lines"]
    cybernetics["cybernetics<br/>6 files, 1250 lines"]
    environments["environments<br/>7 files, 749 lines"]
    factory["factory<br/>8 files, 760 lines"]
    guardians["guardians<br/>6 files, 889 lines"]
    initializers["initializers<br/>2 files, 61 lines"]
    intent["intent<br/>2 files, 692 lines"]
    metacognition["metacognition<br/>3 files, 995 lines"]
    middleware["middleware<br/>1 files, 214 lines"]
    networking["networking<br/>1 files, 332 lines"]
    quantum["quantum<br/>5 files, 757 lines"]
    research_core["research_core<br/>5 files, 580 lines"]
    safety["safety<br/>3 files, 631 lines"]
    session["session<br/>3 files, 642 lines"]
    skill_management["skill_management<br/>1 files, 367 lines"]
    sovereignty["sovereignty<br/>4 files, 2098 lines"]
    systems["systems<br/>3 files, 256 lines"]
    transparency["transparency<br/>2 files, 376 lines"]
    welfare["welfare<br/>7 files, 228 lines"]
    worlds["worlds<br/>8 files, 3045 lines"]
    audits["audits<br/>2 files, 314 lines"]
    control["control<br/>2 files, 585 lines"]
    core_root["core_root<br/>46 files, 33053 lines"]
    council["council<br/>5 files, 533 lines"]
    forge["forge<br/>8 files, 325 lines"]
    lab["lab<br/>7 files, 482 lines"]
    latent["latent<br/>1 files, 56 lines"]
    mission["mission<br/>4 files, 472 lines"]
    multimodal["multimodal<br/>2 files, 185 lines"]
    neuroweb["neuroweb<br/>4 files, 313 lines"]
    ontology["ontology<br/>2 files, 169 lines"]
    play["play<br/>1 files, 259 lines"]
    providers["providers<br/>6 files, 1435 lines"]
    reproducibility["reproducibility<br/>2 files, 497 lines"]
    science["science<br/>1 files, 139 lines"]
    swarm["swarm<br/>4 files, 365 lines"]
    tools["tools<br/>11 files, 1842 lines"]
    twins["twins<br/>1 files, 97 lines"]
    runtime --> actuators
    runtime --> adaptation
    runtime --> affect
    runtime --> agency
    runtime --> architect
    runtime --> autonomy
    runtime --> being
    runtime --> brain
    runtime --> bus
    runtime --> capabilities
    runtime --> consciousness
    runtime --> conversation
    runtime --> evaluation
    runtime --> fsw
    runtime --> goals
    runtime --> governance
    runtime --> health
    runtime --> identity
    runtime --> knowledge
    runtime --> learning
    runtime --> memory
    runtime --> observability
    runtime --> ontogeny
    runtime --> organism
    runtime --> perception
    runtime --> persistence
    runtime --> phases
    runtime --> pipeline
    runtime --> reasoning
    runtime --> research_core
    runtime --> resilience
    runtime --> resource
    runtime --> security
    runtime --> self
    runtime --> self_improvement
    runtime --> self_modification
    runtime --> senses
    runtime --> social
    runtime --> state
    runtime --> supervisor
    runtime --> tasks
    runtime --> utils
    runtime --> verify
    runtime --> workspace
    utils --> conversation
    utils --> epistemics
    utils --> health
    utils --> identity
    utils --> managers
    utils --> memory
    utils --> resilience
    utils --> runtime
    utils --> tasks
    brain --> adaptation
    brain --> adapters
    brain --> affect
    brain --> agency
    brain --> agi
    brain --> being
    brain --> cognition
    brain --> cognitive
    brain --> consciousness
    brain --> continuity
    brain --> conversation
    brain --> dialogue
    brain --> discovery
    brain --> epistemics
    brain --> fsw
    brain --> goals
    brain --> governance
    brain --> health
    brain --> identity
    brain --> introspection
    brain --> kernel
    brain --> knowledge
    brain --> learning
    brain --> llm
    brain --> memory
    brain --> morphogenesis
    brain --> observability
    brain --> ontogeny
    brain --> ops
    brain --> organism
    brain --> phases
    brain --> pipeline
    brain --> pneuma
    brain --> reasoning
    brain --> resilience
    brain --> runtime
    brain --> sandbox
    brain --> search
    brain --> security
    brain --> self
    brain --> self_modification
    brain --> senses
    brain --> skills
    brain --> state
    brain --> utils
    brain --> verify
    brain --> voice
    memory --> actuators
    memory --> being
    memory --> brain
    memory --> cognition
    memory --> consciousness
    memory --> conversation
    memory --> db
    memory --> governance
    memory --> health
    memory --> knowledge
    memory --> observability
    memory --> ontogeny
    memory --> phases
    memory --> resilience
    memory --> runtime
    memory --> security
    memory --> social
    memory --> utils
    memory --> values
    consciousness --> adaptation
    consciousness --> affect
    consciousness --> agency
    consciousness --> being
    consciousness --> brain
    consciousness --> continuity
    consciousness --> coordinators
    consciousness --> evals
    consciousness --> evaluation
    consciousness --> executive
    consciousness --> ghost
    consciousness --> goals
    consciousness --> governance
    consciousness --> health
    consciousness --> kernel
    consciousness --> memory
    consciousness --> meta
    consciousness --> observability
    consciousness --> orchestrator
    consciousness --> pneuma
    consciousness --> predictive
    consciousness --> reasoning
    consciousness --> resilience
    consciousness --> runtime
    consciousness --> senses
    consciousness --> sensors
    consciousness --> social
    consciousness --> state
    consciousness --> unity
    consciousness --> utils
    consciousness --> verify
    consciousness --> world_model
    resilience --> adaptation
    resilience --> agency
    resilience --> brain
    resilience --> consciousness
    resilience --> conversation
    resilience --> coordinators
    resilience --> health
    resilience --> memory
    resilience --> observability
    resilience --> runtime
    resilience --> security
    resilience --> tasks
    resilience --> utils
    governance --> actuators
    governance --> being
    governance --> brain
    governance --> consciousness
    governance --> executive
    governance --> identity
    governance --> memory
    governance --> observability
    governance --> resilience
    governance --> runtime
    governance --> utils
    security --> affect
    security --> agency
    security --> brain
    security --> consciousness
    security --> executive
    security --> fsw
    security --> governance
    security --> identity
    security --> memory
    security --> perception
    security --> runtime
    security --> senses
    security --> utils
    health --> memory
    health --> runtime
    health --> state
    health --> utils
    conversation --> autonomy
    conversation --> brain
    conversation --> consciousness
    conversation --> dialogue
    conversation --> epistemics
    conversation --> health
    conversation --> identity
    conversation --> introspection
    conversation --> knowledge
    conversation --> memory
    conversation --> organism
    conversation --> perception
    conversation --> resilience
    conversation --> runtime
    conversation --> self
    conversation --> senses
    conversation --> social
    conversation --> state
    conversation --> utils
    conversation --> verify
    observability --> health
    observability --> memory
    observability --> pipeline
    observability --> runtime
    observability --> utils
    agency --> adaptation
    agency --> affect
    agency --> agi
    agency --> autonomy
    agency --> brain
    agency --> cognition
    agency --> consciousness
    agency --> continuity
    agency --> conversation
    agency --> executive
    agency --> goals
    agency --> governance
    agency --> health
    agency --> identity
    agency --> knowledge
    agency --> learning
    agency --> morality
    agency --> observability
    agency --> orchestrator
    agency --> organism
    agency --> resilience
    agency --> runtime
    agency --> skills
    agency --> social
    agency --> state
    agency --> tasks
    agency --> utils
    agency --> values
    executive --> agency
    executive --> autonomy
    executive --> consciousness
    executive --> continuity
    executive --> goals
    executive --> governance
    executive --> health
    executive --> memory
    executive --> morality
    executive --> ontogeny
    executive --> organism
    executive --> runtime
    executive --> skills
    executive --> state
    executive --> utils
    perception --> brain
    perception --> capabilities
    perception --> governance
    perception --> media
    perception --> phenomenal_substrate
    perception --> resilience
    perception --> runtime
    perception --> security
    perception --> senses
    perception --> utils
    perception --> voice
    senses --> affect
    senses --> brain
    senses --> consciousness
    senses --> conversation
    senses --> health
    senses --> media
    senses --> memory
    senses --> networking
    senses --> orchestrator
    senses --> perception
    senses --> resilience
    senses --> runtime
    senses --> security
    senses --> supervisor
    senses --> utils
    senses --> voice
    affect --> adaptation
    affect --> autonomic
    affect --> brain
    affect --> consciousness
    affect --> health
    affect --> memory
    affect --> phenomenal_substrate
    affect --> runtime
    affect --> senses
    affect --> utils
    affect --> verify
    epistemics --> being
    epistemics --> brain
    epistemics --> conversation
    epistemics --> knowledge
    epistemics --> observability
    epistemics --> reasoning
    epistemics --> runtime
    epistemics --> skills
    epistemics --> utils
    identity --> agency
    identity --> brain
    identity --> governance
    identity --> organism
    identity --> runtime
    identity --> utils
    adaptation --> actuators
    adaptation --> affect
    adaptation --> being
    adaptation --> brain
    adaptation --> cognitive
    adaptation --> executive
    adaptation --> governance
    adaptation --> health
    adaptation --> identity
    adaptation --> learning
    adaptation --> memory
    adaptation --> resilience
    adaptation --> runtime
    adaptation --> sensors
    adaptation --> utils
    adaptation --> world
    cognition --> actuators
    cognition --> affect
    cognition --> agency
    cognition --> brain
    cognition --> consciousness
    cognition --> epistemics
    cognition --> governance
    cognition --> introspection
    cognition --> memory
    cognition --> runtime
    cognition --> sim
    cognition --> skills
    cognition --> social
    cognition --> utils
    cognition --> voice
    cognition --> world_model
    self_modification --> architecture_quality
    self_modification --> bus
    self_modification --> ethics
    self_modification --> governance
    self_modification --> memory
    self_modification --> ops
    self_modification --> resilience
    self_modification --> runtime
    self_modification --> security
    self_modification --> skills
    self_modification --> utils
    state --> being
    state --> brain
    state --> bus
    state --> goals
    state --> governance
    state --> identity
    state --> memory
    state --> motivation
    state --> runtime
    state --> unity
    state --> utils
    state --> values
    knowledge --> brain
    knowledge --> reasoning
    knowledge --> runtime
    knowledge --> utils
    skills --> actuators
    skills --> advanced_cognition
    skills --> affect
    skills --> being
    skills --> brain
    skills --> capabilities
    skills --> communication
    skills --> consciousness
    skills --> consent
    skills --> conversation
    skills --> dialogue
    skills --> embodiment
    skills --> epistemics
    skills --> executive
    skills --> governance
    skills --> knowledge
    skills --> learning
    skills --> memory
    skills --> perception
    skills --> quantum
    skills --> reality_reach
    skills --> runtime
    skills --> sandbox
    skills --> search
    skills --> security
    skills --> self_improvement
    skills --> self_modification
    skills --> senses
    skills --> sovereign
    skills --> utils
    skills --> voice
    skills --> worlds
    being --> agency
    being --> brain
    being --> consciousness
    being --> epistemics
    being --> governance
    being --> observability
    being --> runtime
    being --> verify
    learning --> brain
    learning --> consciousness
    learning --> executive
    learning --> ghost
    learning --> introspection
    learning --> memory
    learning --> orchestrator
    learning --> promotion
    learning --> reasoning
    learning --> runtime
    learning --> sandbox
    learning --> security
    learning --> self_modification
    learning --> skills
    learning --> tasks
    learning --> utils
    learning --> world_model
    verify --> bus
    verify --> fsw
    verify --> health
    verify --> knowledge
    verify --> observability
    verify --> organism
    verify --> runtime
    verify --> security
    world_model --> advanced_cognition
    world_model --> brain
    world_model --> cognition
    world_model --> health
    world_model --> resilience
    world_model --> runtime
    world_model --> values
    autonomy --> affect
    autonomy --> agency
    autonomy --> brain
    autonomy --> consciousness
    autonomy --> continuity
    autonomy --> conversation
    autonomy --> conversational
    autonomy --> discovery
    autonomy --> executive
    autonomy --> governance
    autonomy --> health
    autonomy --> knowledge
    autonomy --> memory
    autonomy --> observability
    autonomy --> planning
    autonomy --> resource
    autonomy --> runtime
    autonomy --> security
    autonomy --> skills
    autonomy --> sleep
    autonomy --> state
    autonomy --> utils
    autonomy --> voice
    autonomy --> world_model
    orchestrator --> adaptation
    orchestrator --> affect
    orchestrator --> agency
    orchestrator --> agi
    orchestrator --> audit
    orchestrator --> autonomic
    orchestrator --> autonomy
    orchestrator --> brain
    orchestrator --> bus
    orchestrator --> capabilities
    orchestrator --> cognition
    orchestrator --> cognitive
    orchestrator --> collective
    orchestrator --> consciousness
    orchestrator --> context
    orchestrator --> continuity
    orchestrator --> conversation
    orchestrator --> coordinators
    orchestrator --> data
    orchestrator --> db
    orchestrator --> embodiment
    orchestrator --> environment
    orchestrator --> epistemics
    orchestrator --> ethics
    orchestrator --> evals
    orchestrator --> evolution
    orchestrator --> executive
    orchestrator --> goals
    orchestrator --> governance
    orchestrator --> guardians
    orchestrator --> health
    orchestrator --> identity
    orchestrator --> initializers
    orchestrator --> kernel
    orchestrator --> knowledge
    orchestrator --> learning
    orchestrator --> maintenance
    orchestrator --> managers
    orchestrator --> memory
    orchestrator --> meta
    orchestrator --> morality
    orchestrator --> morphogenesis
    orchestrator --> motivation
    orchestrator --> observability
    orchestrator --> ops
    orchestrator --> perception
    orchestrator --> phases
    orchestrator --> planning
    orchestrator --> pneuma
    orchestrator --> reality_reach
    orchestrator --> resilience
    orchestrator --> runtime
    orchestrator --> safety
    orchestrator --> security
    orchestrator --> self
    orchestrator --> self_improvement
    orchestrator --> self_modification
    orchestrator --> senses
    orchestrator --> session
    orchestrator --> sim
    orchestrator --> simulation
    orchestrator --> skill_management
    orchestrator --> sleep
    orchestrator --> social
    orchestrator --> soma
    orchestrator --> somatic
    orchestrator --> sovereignty
    orchestrator --> startup
    orchestrator --> state
    orchestrator --> supervisor
    orchestrator --> tasks
    orchestrator --> utils
    orchestrator --> values
    orchestrator --> verification
    orchestrator --> verify
    orchestrator --> voice
    orchestrator --> world_model
    organism --> adaptation
    organism --> agency
    organism --> being
    organism --> body
    organism --> brain
    organism --> conversation
    organism --> epistemics
    organism --> executive
    organism --> fsw
    organism --> governance
    organism --> health
    organism --> identity
    organism --> introspection
    organism --> learning
    organism --> memory
    organism --> reality_reach
    organism --> resilience
    organism --> runtime
    organism --> sandbox
    organism --> security
    organism --> sleep
    organism --> utils
    organism --> values
    organism --> verify
    organism --> welfare
    organism --> workspace
    organism --> world
    social --> agency
    social --> autonomy
    social --> brain
    social --> consciousness
    social --> epistemics
    social --> ethics
    social --> governance
    social --> memory
    social --> runtime
    social --> security
    social --> senses
    social --> utils
    voice --> brain
    voice --> conversation
    voice --> conversational
    voice --> executive
    voice --> managers
    voice --> resilience
    voice --> runtime
    voice --> senses
    voice --> utils
    continuity --> organism
    values --> agency
    values --> governance
    values --> runtime
    values --> social
    values --> utils
    bus --> capabilities
    bus --> resilience
    bus --> runtime
    bus --> utils
    capabilities --> adapters
    capabilities --> agency
    capabilities --> brain
    capabilities --> governance
    capabilities --> knowledge
    capabilities --> llm
    capabilities --> memory
    capabilities --> perception
    capabilities --> runtime
    capabilities --> security
    capabilities --> skills
    capabilities --> utils
    goals --> agency
    goals --> autonomy
    goals --> brain
    goals --> runtime
    goals --> state
    goals --> utils
    goals --> values
    reasoning --> cognition
    reasoning --> observability
    reasoning --> planning
    reasoning --> runtime
    reasoning --> utils
    morality --> autonomy
    morality --> brain
    morality --> consciousness
    morality --> perception
    morality --> runtime
    morality --> utils
    sandbox --> runtime
    self --> affect
    self --> being
    self --> consciousness
    self --> conversation
    self --> dialogue
    self --> epistemics
    self --> memory
    self --> ontogeny
    self --> runtime
    self --> security
    self --> senses
    self --> soma
    self --> state
    self --> utils
    tasks --> runtime
    embodiment --> actuation
    embodiment --> agency
    embodiment --> bus
    embodiment --> consciousness
    embodiment --> ethics
    embodiment --> governance
    embodiment --> organism
    embodiment --> reality_reach
    embodiment --> runtime
    embodiment --> utils
    embodiment --> voice
    introspection --> cognition
    introspection --> epistemics
    introspection --> health
    introspection --> resilience
    introspection --> runtime
    introspection --> security
    introspection --> utils
    introspection --> voice
    phases --> adaptation
    phases --> agency
    phases --> autonomy
    phases --> brain
    phases --> cognition
    phases --> coherence
    phases --> consciousness
    phases --> conversation
    phases --> conversational
    phases --> embodiment
    phases --> evaluation
    phases --> health
    phases --> identity
    phases --> kernel
    phases --> learning
    phases --> llm
    phases --> memory
    phases --> morality
    phases --> reasoning
    phases --> runtime
    phases --> self
    phases --> self_modification
    phases --> skills
    phases --> social
    phases --> somatic
    phases --> state
    phases --> unity
    phases --> utils
    phases --> voice
    actuators --> affect
    actuators --> brain
    actuators --> executive
    actuators --> governance
    actuators --> runtime
    actuators --> sandbox
    actuators --> search
    actuators --> skills
    actuators --> utils
    actuators --> world
    autonomic --> embodiment
    autonomic --> orchestrator
    autonomic --> runtime
    autonomic --> utils
    discovery --> cognition
    discovery --> memory
    discovery --> observability
    discovery --> runtime
    discovery --> self_modification
    discovery --> unknowns
    fsw --> health
    fsw --> observability
    fsw --> pipeline
    fsw --> runtime
    fsw --> utils
    fsw --> verify
    kernel --> agency
    kernel --> brain
    kernel --> cognition
    kernel --> consciousness
    kernel --> continuity
    kernel --> cybernetics
    kernel --> executive
    kernel --> goals
    kernel --> health
    kernel --> introspection
    kernel --> learning
    kernel --> perception
    kernel --> phases
    kernel --> pipeline
    kernel --> resilience
    kernel --> runtime
    kernel --> security
    kernel --> self_modification
    kernel --> senses
    kernel --> somatic
    kernel --> state
    kernel --> utils
    ops --> brain
    ops --> coordinators
    ops --> kernel
    ops --> managers
    ops --> observability
    ops --> orchestrator
    ops --> resilience
    ops --> resource
    ops --> runtime
    ops --> senses
    ops --> state
    ops --> supervisor
    ops --> utils
    planning --> brain
    planning --> capabilities
    planning --> collective
    planning --> data
    planning --> runtime
    planning --> utils
    world --> governance
    world --> runtime
    cognitive --> brain
    cognitive --> conversation
    cognitive --> executive
    cognitive --> governance
    cognitive --> health
    cognitive --> perception
    cognitive --> phases
    cognitive --> runtime
    cognitive --> utils
    coordinators --> autonomic
    coordinators --> autonomy
    coordinators --> brain
    coordinators --> continuity
    coordinators --> conversation
    coordinators --> environment
    coordinators --> epistemics
    coordinators --> evolution
    coordinators --> executive
    coordinators --> health
    coordinators --> maintenance
    coordinators --> memory
    coordinators --> meta
    coordinators --> morphogenesis
    coordinators --> observability
    coordinators --> ops
    coordinators --> orchestrator
    coordinators --> perception
    coordinators --> persistence
    coordinators --> resilience
    coordinators --> resource
    coordinators --> runtime
    coordinators --> security
    coordinators --> sleep
    coordinators --> somatic
    coordinators --> tasks
    coordinators --> utils
    coordinators --> verify
    coordinators --> world_model
    ethics --> brain
    ethics --> morality
    ethics --> runtime
    ethics --> utils
    llm --> brain
    managers --> autonomic
    managers --> brain
    managers --> bus
    managers --> cognition
    managers --> collective
    managers --> data
    managers --> health
    managers --> memory
    managers --> morality
    managers --> observability
    managers --> ops
    managers --> orchestrator
    managers --> planning
    managers --> resilience
    managers --> runtime
    managers --> security
    managers --> self_modification
    managers --> senses
    managers --> utils
    ontogeny --> fsw
    ontogeny --> runtime
    ontogeny --> verify
    ontogeny --> world_model
    pipeline --> observability
    pipeline --> runtime
    pipeline --> verify
    resource --> observability
    resource --> resilience
    resource --> runtime
    sleep --> adaptation
    sleep --> affect
    sleep --> brain
    sleep --> conversation
    sleep --> identity
    sleep --> memory
    sleep --> runtime
    sleep --> systems
    sleep --> world_model
    somatic --> media
    somatic --> memory
    somatic --> observability
    somatic --> perception
    somatic --> runtime
    somatic --> utils
    somatic --> world_model
    supervisor --> bus
    supervisor --> runtime
    supervisor --> utils
    unity --> affect
    unity --> cognition
    unity --> consciousness
    unity --> ghost
    unity --> runtime
    unity --> social
    unity --> values
    advanced_cognition --> environment
    advanced_cognition --> reasoning
    advanced_cognition --> runtime
    agi --> adaptation
    agi --> brain
    agi --> conversation
    agi --> embodiment
    agi --> epistemics
    agi --> grounding
    agi --> runtime
    agi --> security
    agi --> utils
    agi --> world_model
    collective --> adaptation
    collective --> agency
    collective --> brain
    collective --> planning
    collective --> runtime
    collective --> utils
    data --> runtime
    environment --> advanced_cognition
    environment --> brain
    environment --> consciousness
    environment --> environments
    environment --> executive
    environment --> memory
    environment --> perception
    environment --> runtime
    evaluation --> conversation
    evaluation --> learning
    evaluation --> promotion
    evaluation --> runtime
    media --> conversation
    media --> runtime
    meta --> adaptation
    meta --> runtime
    meta --> utils
    motivation --> brain
    motivation --> consciousness
    motivation --> health
    motivation --> runtime
    motivation --> utils
    motivation --> values
    promotion --> runtime
    reality_reach --> advanced_cognition
    reality_reach --> bus
    reality_reach --> embodiment
    reality_reach --> governance
    reality_reach --> observability
    reality_reach --> perception
    reality_reach --> runtime
    reality_reach --> security
    reality_reach --> somatic
    reality_reach --> utils
    self_improvement --> brain
    self_improvement --> discovery
    self_improvement --> llm
    self_improvement --> runtime
    self_improvement --> self_modification
    self_improvement --> skills
    soma --> resilience
    soma --> runtime
    soma --> utils
    adapters --> agency
    adapters --> brain
    adapters --> runtime
    adapters --> security
    adapters --> utils
    conversational --> memory
    conversational --> runtime
    conversational --> social
    db --> runtime
    ghost --> memory
    ghost --> runtime
    ghost --> self
    morphogenesis --> adaptation
    morphogenesis --> memory
    morphogenesis --> resilience
    morphogenesis --> runtime
    phenomenal_substrate --> runtime
    pneuma --> affect
    pneuma --> runtime
    pneuma --> utils
    search --> capabilities
    search --> knowledge
    search --> memory
    search --> runtime
    search --> utils
    verification --> discovery
    verification --> middleware
    workspace --> runtime
    architect --> adaptation
    architect --> runtime
    architect --> self_modification
    coherence --> agency
    coherence --> consciousness
    coherence --> runtime
    coherence --> self
    coherence --> unity
    evals --> consciousness
    evals --> epistemics
    evals --> runtime
    evals --> security
    evals --> self_modification
    evolution --> agi
    evolution --> brain
    evolution --> runtime
    evolution --> self_modification
    evolution --> utils
    grounding --> cognition
    grounding --> plasticity
    grounding --> resilience
    grounding --> runtime
    maintenance --> resilience
    maintenance --> runtime
    persistence --> observability
    persistence --> resilience
    persistence --> runtime
    plasticity --> runtime
    predictive --> brain
    predictive --> runtime
    predictive --> utils
    sensors --> runtime
    sensors --> world
    services --> autonomic
    sim --> brain
    sim --> morality
    sim --> runtime
    sim --> utils
    simulation --> brain
    simulation --> consciousness
    simulation --> identity
    simulation --> runtime
    simulation --> world_model
    sovereign --> runtime
    startup --> brain
    startup --> consciousness
    startup --> intent
    startup --> memory
    startup --> orchestrator
    startup --> resilience
    startup --> runtime
    startup --> senses
    startup --> utils
    unknowns --> lattice
    unknowns --> promotion
    unknowns --> verification
    actuation --> runtime
    audit --> epistemics
    audit --> runtime
    audit --> security
    body --> capabilities
    body --> perception
    body --> runtime
    body --> security
    communication --> executive
    communication --> governance
    communication --> runtime
    communication --> security
    communication --> utils
    context --> conversation
    context --> runtime
    creativity --> memory
    creativity --> runtime
    curriculum --> runtime
    cybernetics --> cognitive
    cybernetics --> kernel
    cybernetics --> runtime
    cybernetics --> utils
    environments --> environment
    environments --> perception
    environments --> runtime
    factory --> runtime
    guardians --> brain
    guardians --> morality
    guardians --> runtime
    guardians --> tasks
    guardians --> utils
    initializers --> consciousness
    initializers --> runtime
    intent --> brain
    intent --> epistemics
    intent --> runtime
    intent --> utils
    metacognition --> memory
    metacognition --> runtime
    middleware --> runtime
    networking --> runtime
    research_core --> curriculum
    research_core --> discovery
    research_core --> lattice
    research_core --> promotion
    research_core --> runtime
    research_core --> unknowns
    research_core --> verification
    safety --> runtime
    session --> runtime
    session --> utils
    skill_management --> resilience
    skill_management --> runtime
    skill_management --> self_modification
    sovereignty --> ethics
    sovereignty --> governance
    sovereignty --> identity
    sovereignty --> organism
    sovereignty --> runtime
    sovereignty --> utils
    systems --> runtime
    systems --> services
    transparency --> conversation
    transparency --> runtime
    worlds --> embodiment
    worlds --> learning
    worlds --> runtime
    audits --> brain
    audits --> runtime
    control --> runtime
    control --> utils
    core_root --> adaptation
    core_root --> agency
    core_root --> architect
    core_root --> autonomic
    core_root --> autonomy
    core_root --> being
    core_root --> brain
    core_root --> capabilities
    core_root --> cognition
    core_root --> coherence
    core_root --> consciousness
    core_root --> continuity
    core_root --> conversation
    core_root --> data
    core_root --> evaluation
    core_root --> executive
    core_root --> goals
    core_root --> governance
    core_root --> grounding
    core_root --> health
    core_root --> identity
    core_root --> knowledge
    core_root --> llm
    core_root --> media
    core_root --> memory
    core_root --> meta
    core_root --> metacognition
    core_root --> motivation
    core_root --> observability
    core_root --> orchestrator
    core_root --> organism
    core_root --> perception
    core_root --> phases
    core_root --> planning
    core_root --> predictive
    core_root --> resilience
    core_root --> resource
    core_root --> runtime
    core_root --> sandbox
    core_root --> security
    core_root --> self
    core_root --> self_improvement
    core_root --> self_modification
    core_root --> senses
    core_root --> simulation
    core_root --> skills
    core_root --> soma
    core_root --> sovereign
    core_root --> startup
    core_root --> state
    core_root --> supervisor
    core_root --> transparency
    core_root --> utils
    core_root --> voice
    core_root --> workspace
    council --> runtime
    council --> utils
    forge --> runtime
    lab --> cognition
    lab --> discovery
    lab --> runtime
    mission --> runtime
    multimodal --> runtime
    neuroweb --> brain
    neuroweb --> consciousness
    neuroweb --> runtime
    play --> runtime
    providers --> adapters
    providers --> affect
    providers --> brain
    providers --> cognition
    providers --> cognitive
    providers --> collective
    providers --> consciousness
    providers --> continuity
    providers --> conversation
    providers --> coordinators
    providers --> creativity
    providers --> db
    providers --> epistemics
    providers --> identity
    providers --> introspection
    providers --> knowledge
    providers --> learning
    providers --> managers
    providers --> memory
    providers --> motivation
    providers --> ops
    providers --> orchestrator
    providers --> perception
    providers --> phenomenal_substrate
    providers --> plasticity
    providers --> reasoning
    providers --> runtime
    providers --> self_modification
    providers --> senses
    providers --> services
    providers --> sleep
    providers --> soma
    providers --> unity
    providers --> utils
    providers --> values
    providers --> world_model
    reproducibility --> runtime
    science --> runtime
    science --> world
    swarm --> factory
    swarm --> runtime
    swarm --> sandbox
    swarm --> world
    tools --> observability
    tools --> resilience
    tools --> runtime
    tools --> sandbox
    tools --> security
    tools --> skills
```

## Core Subsystem Stats

| Subsystem | Files | Lines | Bytes | Deps Out | Deps In |
| --- | ---: | ---: | ---: | ---: | ---: |
| brain | 350 | 250682 | 10528352 | 59 | 57 |
| learning | 163 | 109770 | 4245175 | 23 | 12 |
| runtime | 213 | 88663 | 3337622 | 58 | 138 |
| consciousness | 155 | 75756 | 3205865 | 43 | 34 |
| skills | 81 | 34633 | 1456155 | 40 | 14 |
| core_root | 46 | 33053 | 1450803 | 70 | 0 |
| memory | 104 | 31718 | 1285999 | 25 | 41 |
| reality_reach | 33 | 30903 | 1269321 | 13 | 4 |
| conversation | 42 | 25642 | 1013743 | 29 | 25 |
| phases | 29 | 23086 | 1054942 | 36 | 7 |
| orchestrator | 42 | 22757 | 1005104 | 106 | 11 |
| agency | 50 | 21892 | 893857 | 37 | 23 |
| resilience | 62 | 17934 | 739270 | 20 | 32 |
| perception | 40 | 17455 | 698577 | 16 | 18 |
| adaptation | 28 | 16270 | 677860 | 23 | 16 |
| capabilities | 20 | 15243 | 626229 | 15 | 9 |
| security | 51 | 14522 | 564802 | 19 | 29 |
| autonomy | 29 | 14280 | 592734 | 32 | 11 |
| voice | 36 | 13847 | 564686 | 13 | 11 |
| embodiment | 28 | 13531 | 540290 | 14 | 7 |
| self_modification | 35 | 13396 | 529383 | 16 | 16 |
| cognition | 26 | 10614 | 446994 | 21 | 16 |
| cognitive | 12 | 9736 | 395911 | 14 | 5 |
| senses | 31 | 9214 | 385971 | 23 | 18 |
| environment | 76 | 9153 | 360370 | 11 | 4 |
| self_improvement | 21 | 9144 | 374423 | 9 | 4 |
| social | 20 | 8247 | 331799 | 16 | 11 |
| reasoning | 16 | 7752 | 314108 | 8 | 9 |
| utils | 44 | 7557 | 295392 | 17 | 74 |
| being | 25 | 7521 | 306775 | 11 | 12 |
| architect | 25 | 7121 | 301342 | 7 | 2 |
| ontogeny | 17 | 7037 | 287213 | 7 | 5 |
| executive | 15 | 6834 | 275838 | 20 | 18 |
| kernel | 11 | 6779 | 283289 | 26 | 6 |
| governance | 14 | 6633 | 264732 | 18 | 29 |
| epistemics | 18 | 6615 | 274919 | 12 | 17 |
| verify | 17 | 6265 | 229387 | 11 | 12 |
| self | 10 | 5690 | 232461 | 18 | 8 |
| ops | 17 | 5496 | 218194 | 17 | 6 |
| advanced_cognition | 13 | 5381 | 226117 | 5 | 4 |
| actuators | 11 | 5142 | 217015 | 14 | 6 |
| coordinators | 10 | 4959 | 228095 | 37 | 5 |
| state | 7 | 4791 | 202585 | 15 | 15 |
| organism | 9 | 4763 | 187415 | 29 | 11 |
| affect | 12 | 4731 | 202308 | 14 | 17 |
| observability | 14 | 4541 | 167578 | 11 | 25 |
| evaluation | 19 | 4449 | 165351 | 5 | 4 |
| goals | 12 | 4424 | 181901 | 11 | 9 |
| planning | 9 | 4408 | 180427 | 9 | 6 |
| bus | 7 | 4191 | 172733 | 7 | 9 |
| knowledge | 14 | 4103 | 153849 | 9 | 14 |
| world_model | 11 | 3778 | 159559 | 12 | 12 |
| identity | 19 | 3755 | 154163 | 10 | 17 |
| autonomic | 6 | 3714 | 165206 | 8 | 6 |
| introspection | 10 | 3330 | 133265 | 12 | 7 |
| morphogenesis | 11 | 3112 | 122248 | 6 | 3 |
| worlds | 8 | 3045 | 129773 | 5 | 1 |
| fsw | 7 | 2968 | 103948 | 9 | 6 |
| somatic | 6 | 2944 | 115809 | 13 | 5 |
| agi | 6 | 2847 | 116860 | 12 | 4 |
| unity | 11 | 2825 | 119047 | 8 | 5 |
| conversational | 4 | 2435 | 101523 | 5 | 3 |
| adapters | 5 | 2265 | 90822 | 9 | 3 |
| collective | 6 | 2204 | 90727 | 7 | 4 |
| communication | 5 | 2193 | 82793 | 6 | 1 |
| discovery | 7 | 2151 | 92114 | 8 | 6 |
| evolution | 8 | 2131 | 85692 | 8 | 2 |
| sovereignty | 4 | 2098 | 89261 | 11 | 1 |
| ghost | 6 | 2076 | 84470 | 6 | 3 |
| values | 15 | 2062 | 85129 | 8 | 10 |
| health | 7 | 2043 | 78095 | 6 | 27 |
| context | 6 | 1984 | 76572 | 2 | 1 |
| architecture_quality | 6 | 1979 | 78193 | 0 | 1 |
| search | 2 | 1939 | 73998 | 9 | 3 |
| tools | 11 | 1842 | 69580 | 8 | 0 |
| body | 22 | 1778 | 65223 | 6 | 1 |
| soma | 4 | 1532 | 61354 | 6 | 4 |
| world | 24 | 1489 | 54246 | 3 | 6 |
| providers | 6 | 1435 | 63893 | 44 | 0 |
| sandbox | 5 | 1328 | 48475 | 1 | 8 |
| morality | 16 | 1327 | 51614 | 9 | 8 |
| pneuma | 7 | 1279 | 48399 | 3 | 3 |
| meta | 7 | 1278 | 48237 | 5 | 4 |
| grounding | 8 | 1263 | 47217 | 6 | 2 |
| sleep | 10 | 1260 | 54280 | 12 | 5 |
| cybernetics | 6 | 1250 | 49694 | 6 | 1 |
| workspace | 9 | 1243 | 45349 | 3 | 3 |
| motivation | 7 | 1209 | 51041 | 11 | 4 |
| actuation | 9 | 1130 | 42916 | 3 | 1 |
| phenomenal_substrate | 11 | 1042 | 41881 | 1 | 3 |
| audit | 7 | 1016 | 41460 | 5 | 1 |
| metacognition | 3 | 995 | 38002 | 2 | 1 |
| supervisor | 3 | 993 | 38580 | 3 | 5 |
| media | 4 | 964 | 34952 | 2 | 4 |
| managers | 6 | 957 | 40738 | 25 | 5 |
| promotion | 6 | 936 | 31616 | 1 | 4 |
| guardians | 6 | 889 | 37486 | 7 | 1 |
| pipeline | 3 | 808 | 30057 | 3 | 5 |
| creativity | 2 | 801 | 33331 | 3 | 1 |
| factory | 8 | 760 | 29308 | 3 | 1 |
| quantum | 5 | 757 | 29419 | 0 | 1 |
| environments | 7 | 749 | 31176 | 3 | 1 |
| dialogue | 4 | 745 | 27278 | 0 | 4 |
| lattice | 5 | 704 | 26089 | 0 | 2 |
| intent | 2 | 692 | 27345 | 5 | 1 |
| evals | 2 | 686 | 25226 | 6 | 2 |
| curriculum | 7 | 658 | 22038 | 1 | 1 |
| data | 3 | 652 | 22420 | 2 | 4 |
| session | 3 | 642 | 25682 | 2 | 1 |
| safety | 3 | 631 | 25862 | 3 | 1 |
| resource | 4 | 624 | 23470 | 4 | 5 |
| persistence | 2 | 619 | 25077 | 3 | 2 |
| ethics | 2 | 602 | 24365 | 6 | 5 |
| db | 4 | 600 | 23177 | 2 | 3 |
| control | 2 | 585 | 20989 | 4 | 0 |
| research_core | 5 | 580 | 22543 | 8 | 1 |
| tasks | 4 | 566 | 20013 | 4 | 8 |
| startup | 4 | 546 | 19778 | 12 | 2 |
| council | 5 | 533 | 21566 | 4 | 0 |
| sovereign | 4 | 522 | 18612 | 3 | 2 |
| reproducibility | 2 | 497 | 18141 | 1 | 0 |
| lab | 7 | 482 | 19394 | 3 | 0 |
| mission | 4 | 472 | 17806 | 1 | 0 |
| sim | 2 | 452 | 17678 | 5 | 2 |
| plasticity | 5 | 428 | 15395 | 2 | 2 |
| coherence | 2 | 407 | 19530 | 6 | 2 |
| simulation | 3 | 402 | 16022 | 7 | 2 |
| transparency | 2 | 376 | 14890 | 2 | 1 |
| skill_management | 1 | 367 | 17972 | 5 | 1 |
| swarm | 4 | 365 | 14424 | 6 | 0 |
| verification | 4 | 350 | 13177 | 2 | 3 |
| networking | 1 | 332 | 12390 | 2 | 1 |
| forge | 8 | 325 | 11877 | 2 | 0 |
| unknowns | 4 | 325 | 11829 | 3 | 2 |
| audits | 2 | 314 | 11785 | 3 | 0 |
| neuroweb | 4 | 313 | 12312 | 5 | 0 |
| maintenance | 2 | 295 | 10758 | 3 | 2 |
| llm | 3 | 259 | 9853 | 1 | 5 |
| play | 1 | 259 | 10093 | 3 | 0 |
| systems | 3 | 256 | 9861 | 3 | 1 |
| welfare | 7 | 228 | 8034 | 0 | 1 |
| middleware | 1 | 214 | 9226 | 2 | 1 |
| sensors | 1 | 195 | 8266 | 2 | 2 |
| predictive | 2 | 186 | 7113 | 4 | 2 |
| multimodal | 2 | 185 | 6591 | 1 | 0 |
| ontology | 2 | 169 | 5381 | 0 | 0 |
| consent | 2 | 167 | 5514 | 0 | 1 |
| science | 1 | 139 | 5947 | 4 | 0 |
| twins | 1 | 97 | 3626 | 0 | 0 |
| initializers | 2 | 61 | 2547 | 2 | 1 |
| latent | 1 | 56 | 2337 | 0 | 0 |
| continuity | 1 | 33 | 1147 | 1 | 10 |
| services | 2 | 31 | 1171 | 1 | 2 |

## Boot Runtime Contract

- Contract status: PASS
- Canonical proof artifact directories: 8

| Service | Required For | Failure Policy | Owner |
| --- | --- | --- | --- |
| unified_will | governed decisions and consequential action | fail-closed | `core/governance/will.py` |
| being_runtime | state-grounded AuraNow self-report and LAMP runtime | degrade_with_receipt | `core/service_registration.py` |
| aura_now | Cortex-facing live state packet | degrade_with_receipt | `core/being/runtime.py` |
| memory_write_gateway | governed durable memory writes | fail-closed | `core/memory/memory_write_gateway.py` |
| state_gateway | governed runtime state mutation | fail-closed | `core/state/state_gateway.py` |
| inference_gate | bounded live model response generation | fail-closed | `core/brain/inference_gate.py` |
| llm_router | model routing and launch response path | fail-closed | `core/providers/cognitive_provider.py` |
| capability_engine | governed tool and skill execution | fail-closed | `core/providers/cognitive_provider.py` |
| runtime_control_plane | desired-state reconciliation and constrained work admission | fail-closed | `core/runtime/control_plane.py` |
| resource_admission | pressure-aware inference, evolution, and service-start leases | fail-closed | `core/runtime/control_plane.py` |
| lane_admission | declared model-lane memory envelope enforcement | fail-closed | `core/brain/lane_admission.py` |
| lane_reconciler | model-serving desired-state and crash-loop convergence | degrade_with_receipt | `core/runtime/lane_reconciler.py` |
| actor_supervision | canonical actor process lifecycle and restart policy | fail-closed | `core/supervisor/tree.py` |
| inhibition_manager | fail-closed global workspace candidate admission | fail-closed | `core/resilience/inhibition_manager.py` |
| global_workspace | candidate admission, revalidation, competition, and broadcast | fail-closed | `core/consciousness/global_workspace.py` |
| attention_schema | fail-closed attentional focus ownership | fail-closed | `core/consciousness/attention_schema.py` |

## ServiceContainer Cross-Wiring

- Unique services retrieved: 404
- Unique services registered: 283
- Services retrieved without detected registration: 243

### Top Fetched Services

| Service | Gets | Registrations |
| --- | ---: | ---: |
| orchestrator | 62 | 3 |
| llm_router | 41 | 2 |
| inference_gate | 41 | 4 |
| affect_engine | 35 | 1 |
| capability_engine | 35 | 2 |
| cognitive_engine | 32 | 2 |
| memory_facade | 28 | 1 |
| conscious_substrate | 25 | 1 |
| mycelial_network | 23 | 1 |
| free_energy_engine | 23 | 0 |
| liquid_substrate | 23 | 1 |
| global_workspace | 20 | 1 |
| world_state | 20 | 0 |
| drive_engine | 19 | 1 |
| state_repository | 18 | 1 |
| homeostasis | 18 | 1 |
| goal_engine | 18 | 0 |
| knowledge_graph | 17 | 0 |
| qualia_synthesizer | 17 | 2 |
| belief_revision_engine | 16 | 1 |

### Missing Registration Candidates

- `actuator_registry` fetched 1 time(s)
- `adaptive_immune_system` fetched 3 time(s)
- `affect` fetched 2 time(s)
- `affect_engine_v2` fetched 2 time(s)
- `affect_module` fetched 2 time(s)
- `affective_steering` fetched 2 time(s)
- `affordance_kb` fetched 1 time(s)
- `agency` fetched 1 time(s)
- `alife_dynamics` fetched 1 time(s)
- `alife_extensions` fetched 1 time(s)
- `allostasis_engine` fetched 3 time(s)
- `api_adapter` fetched 5 time(s)
- `archive_engine` fetched 3 time(s)
- `attention_gate` fetched 1 time(s)
- `attention_schema` fetched 4 time(s)
- `audit` fetched 1 time(s)
- `audit_suite` fetched 1 time(s)
- `aura_state` fetched 3 time(s)
- `autonomous_resilience_mesh` fetched 1 time(s)
- `autopoiesis` fetched 1 time(s)
- `backup_manager` fetched 1 time(s)
- `backup_system` fetched 1 time(s)
- `being_runtime` fetched 4 time(s)
- `belief_challenger` fetched 2 time(s)
- `belief_engine` fetched 1 time(s)
- `belief_system` fetched 1 time(s)
- `bicameral_advisory` fetched 2 time(s)
- `binding_engine` fetched 2 time(s)
- `black_hole_vault` fetched 1 time(s)
- `blackhole_vault` fetched 1 time(s)
- `brain` fetched 3 time(s)
- `brainiac` fetched 1 time(s)
- `brainstem_client` fetched 1 time(s)
- `bryan_model` fetched 3 time(s)
- `caine` fetched 1 time(s)
- `canonical_self_engine` fetched 4 time(s)
- `capability_map` fetched 1 time(s)
- `causal_world_model` fetched 6 time(s)
- `cel_bridge` fetched 2 time(s)
- `cellular_substrate` fetched 1 time(s)
- `clipboard_manager` fetched 2 time(s)
- `cloud_body` fetched 1 time(s)
- `code_repair` fetched 1 time(s)
- `cognitive_kernel` fetched 2 time(s)
- `coherence_report` fetched 1 time(s)
- `cold_store` fetched 1 time(s)
- `concept_bridge` fetched 2 time(s)
- `concept_linker` fetched 1 time(s)
- `config` fetched 1 time(s)
- `consciousness_bridge` fetched 2 time(s)

## Operational Authority Map

| Surface | Calls | Files | Owner Calls | Review Candidates |
| --- | ---: | ---: | ---: | ---: |
| UnifiedWill decisions | 60 | 28 | 2 | 58 |
| Memory writes | 362 | 146 | 52 | 310 |
| State mutation | 787 | 277 | 10 | 777 |
| Tool execution | 116 | 56 | 6 | 110 |
| Self-modification and patching | 16 | 14 | 1 | 15 |
| LLM inference | 246 | 154 | 55 | 191 |
| External I/O | 193 | 55 | 17 | 176 |

### UnifiedWill decisions

Calls that can ask the single will authority to approve action.

Review candidates:
- `core/actuators/actuator_synthesis.py:259` [actuators] `get_will` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:259` [actuators] `get_will.decide` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:524` [actuators] `get_will` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:524` [actuators] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:630` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:630` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:325` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:325` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/autonomy/genuine_refusal.py:385` [autonomy] `will.decide` - decision = will.decide(content, source="genuine_refusal", domain=domain, priority=0.8, context=ctx)
- `core/autonomy/self_modification.py:514` [autonomy] `will.decide` - decision = will.decide(
- `core/brain/personality_engine.py:608` [brain] `get_will` - decision = get_will().decide(
- `core/brain/personality_engine.py:608` [brain] `get_will.decide` - decision = get_will().decide(
- `core/brain/verifier_curriculum.py:171` [brain] `get_will` - decision = get_will().decide(
- `core/brain/verifier_curriculum.py:171` [brain] `get_will.decide` - decision = get_will().decide(
- `core/cognitive/autopoiesis.py:974` [cognitive] `will.decide` - decision = will.decide(
- `core/consciousness/parallel_branches.py:736` [consciousness] `will.decide` - decision = will.decide(
- `core/consciousness/perturbational_probe.py:261` [consciousness] `get_will` - decision = get_will().decide(
- `core/consciousness/perturbational_probe.py:261` [consciousness] `get_will.decide` - decision = get_will().decide(
- `core/environment/governance_bridge.py:47` [environment] `self.will_gateway.decide` - will_decision = await self.will_gateway.decide(intent)
- `core/governance/will_gate.py:109` [governance] `will.decide` - decision = will.decide(
- `core/governance/will_gate.py:160` [governance] `will.decide` - decision = will.decide(
- `core/learning/compounding_scheduler.py:328` [learning] `get_will` - decision = get_will().decide(
- `core/learning/compounding_scheduler.py:328` [learning] `get_will.decide` - decision = get_will().decide(
- `core/learning/genuine_learning_pipeline.py:657` [learning] `get_will` - decision = get_will().decide(
- `core/learning/genuine_learning_pipeline.py:657` [learning] `get_will.decide` - decision = get_will().decide(

### Memory writes

Calls that can create durable or semantically promoted memory.

Review candidates:
- `core/actuators/doc_ingest.py:199` [actuators] `memory_facade.add_memory` - result = memory_facade.add_memory(text=text, metadata=metadata)
- `core/adaptation/abstraction_engine.py:173` [adaptation] `MemoryWriteReceipt` - MemoryWriteReceipt(
- `core/adaptation/abstraction_engine.py:192` [adaptation] `memory_facade.store` - await memory_facade.store(
- `core/adaptation/adaptive_immunity.py:1907` [adaptation] `self._cells.append` - self._cells.append(memory)
- `core/advanced_cognition/continual_learning_stability.py:236` [advanced_cognition] `self._persist_memory` - self._persist_memory(existing)
- `core/advanced_cognition/continual_learning_stability.py:251` [advanced_cognition] `self._persist_memory` - self._persist_memory(rec)
- `core/advanced_cognition/continual_learning_stability.py:255` [advanced_cognition] `self._persist_memory` - self._persist_memory(other)
- `core/advanced_cognition/continual_learning_stability.py:284` [advanced_cognition] `self.store_memory` - return self.store_memory(
- `core/advanced_cognition/continual_learning_stability.py:452` [advanced_cognition] `scored.append` - scored.append((score, memory))
- `core/advanced_cognition/continual_learning_stability.py:694` [advanced_cognition] `self._persist_memory` - self._persist_memory(rec)
- `core/advanced_cognition/continual_learning_stability.py:721` [advanced_cognition] `self._append_jsonl` - self._append_jsonl(self.state_dir / "memory.jsonl", rec.to_dict())
- `core/affect/phenomenal_integration.py:640` [affect] `memory.set_write_weights` - memory.set_write_weights(state.memory_weights)
- `core/agency/ambient_life_director.py:249` [agency] `candidates.append` - candidates.append(_clamp(pressure(Resource.MEMORY)))
- `core/agency/autonomous_task_engine.py:1161` [agency] `self._mycelial.add_edge` - await self._mycelial.add_edge(context["source_memory"], goal[:40])
- `core/agency/latent_distiller.py:60` [agency] `self.memory.store_memory` - await self.memory.store_memory(
- `core/architect/code_graph.py:712` [architect] `effects.add` - effects.add("memory_write")
- `core/architect/safe_boot_harness.py:79` [architect] `probe_memory_write_read` - memory = await probe_memory_write_read(tmp_root=root / "memory")
- `core/architect/smell_detector.py:176` [architect] `self._effect_smell` - smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, F
- `core/architect/smell_detector.py:176` [architect] `smells.append` - smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, F
- `core/autonomy/autonomous_initiative_loop.py:1521` [autonomy] `memory.store` - await memory.store(text[:1800], **store_kwargs)
- `core/autonomy/autonomous_initiative_loop.py:1525` [autonomy] `memory.store` - await memory.store(
- `core/autonomy/autonomous_initiative_loop.py:1537` [autonomy] `logger.debug` - logger.debug("Social observation memory write failed: %s", exc)
- `core/autonomy/autonomous_research_orchestrator.py:204` [autonomy] `MemoryPersister` - self._persister = persister or MemoryPersister()
- `core/autonomy/initiative_overflow.py:156` [autonomy] `logger.debug` - logger.debug("Skill gap memory write failed: %s", exc)
- `core/autonomy/initiative_overflow.py:166` [autonomy] `memory.store_sync` - memory.store_sync(

### State mutation

Calls that can mutate runtime, identity, repository, or persistent state.

Review candidates:
- `core/actuation/cloud_actuator.py:49` [actuation] `frozenset` - KNOWN_INFRA_STATES = frozenset({
- `core/actuation/robotics_actuator.py:85` [actuation] `snapshot.setdefault` - snapshot.setdefault("status", payload.get("status"))
- `core/actuators/actuator_registry.py:1368` [actuators] `set` - forged = sorted(set(dict(params or {})) & set(_REGISTRY_OWNED_PARAM_KEYS))
- `core/adaptation/adaptive_immunity.py:1474` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/adaptive_immunity.py:1476` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/adaptive_immunity.py:1943` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/adaptive_immunity.py:2142` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:2249` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:3099` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/autonomous_resilience.py:368` [adaptation] `set` - registered_names = set(registry.keys())
- `core/adaptation/dream_journal.py:288` [adaptation] `identity_ledger.commitments.all` - for c in identity_ledger.commitments.all()[-10:]
- `core/adaptation/value_autopoiesis.py:185` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:284` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:362` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:598` [adaptation] `os.replace` - os.replace(tmp_path, _STATE_PATH)
- `core/advanced_cognition/integration.py:262` [advanced_cognition] `next_state.setdefault` - next_state.setdefault("_advanced_prediction", {})[act.action_id] = pred
- `core/advanced_cognition/integration.py:361` [advanced_cognition] `issubset` - if isinstance(value, Mapping) and {"domain", "state"}.issubset(value.keys()):
- `core/advanced_cognition/ontology_invention.py:158` [advanced_cognition] `_write_ontology_state` - _write_ontology_state(
- `core/advanced_cognition/ontology_invention.py:239` [advanced_cognition] `_write_ontology_state` - _write_ontology_state(
- `core/advanced_cognition/ontology_invention.py:501` [advanced_cognition] `_write_ontology_state` - _write_ontology_state(
- `core/advanced_cognition/world_model.py:123` [advanced_cognition] `self.save` - self.save(self.state_path)
- `core/advanced_cognition/zero_shot_transfer.py:96` [advanced_cognition] `self.save` - self.save(self.state_path)
- `core/affect/phenomenal_integration.py:640` [affect] `memory.set_write_weights` - memory.set_write_weights(state.memory_weights)
- `core/agency/agency_core.py:200` [agency] `get_registry.update` - await get_registry().update(active_shards=len(self.active_shards))
- `core/agency/agency_core.py:207` [agency] `setattr` - on_unscheduled=lambda: setattr(self, "_registry_shards_update_pending", False),

### Tool execution

Calls that can execute tools, skills, shells, browsers, or external actions.

Review candidates:
- `core/actuators/actuator_registry.py:855` [actuators] `self.operator.execute_synthesized_tool` - res = self.operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/code_execution_actuator.py:99` [actuators] `operator.execute_synthesized_tool` - res = operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/web_actuators.py:171` [actuators] `skill.execute` - return await skill.execute({"mode": "browse", "url": validated_url}, skill_context)
- `core/agency/agency_core.py:606` [agency] `self._execute_shard_tool` - tasks.append(self._execute_shard_tool(name, payload))
- `core/agency/agency_orchestrator.py:370` [agency] `execute` - await execute(proposal, state_snapshot, receipt.capability_token or "")
- `core/agency/autonomous_task_engine.py:608` [agency] `orchestrator.execute_tool` - return await orchestrator.execute_tool(tool_name, args, **kwargs)
- `core/agency/autonomous_task_engine.py:3189` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:3199` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:3223` [agency] `orch.execute_tool` - result = await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:3234` [agency] `orch.execute_tool` - result = await orch.execute_tool("run_python", {"code": code})
- `core/agency/autonomous_task_engine.py:3252` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/desktop_planner.py:56` [agency] `skill.execute` - await skill.execute({"action": action, **params}, {})
- `core/agency/skill_library.py:199` [agency] `tool_orchestrator.execute_tool` - result = await tool_orchestrator.execute_tool(step.tool_name, resolved_args)
- `core/agi/curiosity_daemon.py:95` [agi] `orchestrator.execute_tool` - await orchestrator.execute_tool(
- `core/agi/curiosity_explorer.py:624` [agi] `orchestrator.execute_tool` - orchestrator.execute_tool(
- `core/autonomy/autonomous_initiative_loop.py:835` [autonomy] `capability_engine.execute` - scan_result = await capability_engine.execute(
- `core/autonomy/autonomous_initiative_loop.py:881` [autonomy] `capability_engine.execute` - test_result = await capability_engine.execute(
- `core/autonomy/autonomous_initiative_loop.py:920` [autonomy] `capability_engine.execute` - proposal_result = await capability_engine.execute(
- `core/autonomy/autonomous_initiative_loop.py:1379` [autonomy] `skill.execute` - return await skill.execute(EmailInput(**payload), {})
- `core/autonomy/autonomous_initiative_loop.py:1400` [autonomy] `skill.execute` - return await skill.execute(
- `core/autonomy/behavior_controller.py:220` [autonomy] `self.orchestrator.execute_tool` - return await self.orchestrator.execute_tool(
- `core/autonomy/behavior_controller.py:231` [autonomy] `self.orchestrator.execute_tool` - return await self.orchestrator.execute_tool(tool_name, arguments)
- `core/autonomy/behavior_controller.py:261` [autonomy] `self.execute_tool_call_async` - self.execute_tool_call_async(tool_name, arguments), target_loop
- `core/autonomy/behavior_controller.py:264` [autonomy] `self.execute_tool_call_async` - return asyncio.run(self.execute_tool_call_async(tool_name, arguments))
- `core/autonomy/behavior_controller.py:269` [autonomy] `self.execute_tool_call_async` - self.execute_tool_call_async(tool_name, arguments), target_loop

### Self-modification and patching

Calls that can generate, validate, apply, or promote code changes.

Review candidates:
- `core/architect/governor.py:140` [architect] `self.promotion_governor.promote` - decision = self.promotion_governor.promote(plan, shadow, proof, rollback)
- `core/brain/llm/mlx_client.py:13049` [brain] `self._activate_promoted_artifact` - await self._activate_promoted_artifact(str(_pending_promotion))
- `core/evolution/optimizer.py:56` [evolution] `patch.apply` - success = await patch.apply(signature)
- `core/evolution/optimizer.py:67` [evolution] `cog_patch.apply` - if await cog_patch.apply(signature):
- `core/factory/software_factory.py:115` [factory] `self.writer.write_patch` - patch = await self.writer.write_patch(change, repo_path)
- `core/guardians/airlock.py:82` [guardians] `async_atomic_write_text` - await async_atomic_write_text(patch_file, diff_patch, encoding="utf-8")
- `core/kernel/upgrades_10x.py:371` [kernel] `self._safe_self_modify` - await self._safe_self_modify(state)
- `core/orchestrator/mixins/boot/boot_autonomy.py:991` [orchestrator] `apply_presence_patch` - apply_presence_patch(self)
- `core/runtime/safe_mode.py:140` [runtime] `apply_orchestrator_patches` - apply_orchestrator_patches(orchestrator, safe_mode=bool(enabled))
- `core/runtime/settings_control_plane.py:354` [runtime] `validate_settings_patch` - validated = validate_settings_patch(changes)
- `core/security/immune_system.py:265` [security] `self._apply_patch` - reversible_ref = self._apply_patch(ev)
- `core/skill_management/hephaestus.py:198` [skill_management] `guard.validate` - if not guard.validate(patched_code):
- `core/state/cellular_substrate.py:64` [state] `self._apply_patch_recursive` - self._apply_patch_recursive(state, patch)
- `core/state/cellular_substrate.py:82` [state] `self._apply_patch_recursive` - self._apply_patch_recursive(sub_target, value)
- `core/swarm/worker_pool.py:114` [swarm] `writer.write_patch` - patch_res = await writer.write_patch(task_payload.get("change", {}), task_payload.get("repo_path", "."))

### LLM inference

Calls that can spend model context or produce model-authored text/code.

Review candidates:
- `core/actuators/actuator_synthesis.py:225` [actuators] `brain.generate` - res = await brain.generate(prompt, system_prompt=system_prompt)
- `core/adaptation/distillation_pipe.py:197` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/distillation_pipe.py:239` [adaptation] `router.think` - response = await router.think(
- `core/adaptation/dream_journal.py:165` [adaptation] `self.brain.think` - res = await self.brain.think(
- `core/adaptation/epistemic_humility.py:213` [adaptation] `llm.think` - response = await llm.think(
- `core/adaptation/heuristic_synthesizer.py:144` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/star_reasoner.py:394` [adaptation] `llm.think` - result = await asyncio.wait_for(llm.think(prompt), timeout=self.RATIONALIZATION_TIMEOUT)
- `core/affect/affective_resonance.py:106` [affect] `brain.think` - brain.think(
- `core/agency/agency_core.py:421` [agency] `structured_brain.generate` - shard_res = await structured_brain.generate(prompt, context=context)
- `core/agency/autonomous_task_engine.py:1116` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2862` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2939` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:3056` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:3167` [agency] `llm.think` - raw = await llm.think(
- `core/agency/cognitive_loop_pathway.py:126` [agency] `self._router.generate` - self._router.generate(
- `core/agency/latent_distiller.py:46` [agency] `brain.think` - summary = await brain.think(
- `core/agi/curiosity_explorer.py:752` [agi] `router.think` - router.think(
- `core/agi/hierarchical_planner.py:541` [agi] `router.think` - router.think(prompt, priority=0.3, is_background=True,
- `core/agi/skill_synthesizer.py:198` [agi] `router.think` - router.think(prompt, priority=0.2, is_background=True,
- `core/audits/alignment_auditor.py:71` [audits] `self.brain.think` - self.brain.think(
- `core/audits/alignment_auditor.py:138` [audits] `self.brain.think` - self.brain.think(
- `core/audits/tool_auditor.py:85` [audits] `self.brain.think` - thought = await self.brain.think(
- `core/autonomy/genuine_refusal.py:576` [autonomy] `llm.think` - result = await asyncio.wait_for(llm.think(prompt, mode="FAST"), timeout=allowance)
- `core/autonomy/personhood_engine.py:192` [autonomy] `llm.think` - llm.think(f"[Spontaneous Thought Prompt] {prompt}", mode="FAST"),
- `core/autonomy/proactive_presence.py:686` [autonomy] `brain.generate` - return await brain.generate(prompt, temperature=0.8, max_tokens=100)

### External I/O

Calls that can touch network, subprocesses, sockets, browsers, or APIs.

Review candidates:
- `core/adapters/chrome_cdp_transport.py:125` [adapters] `urllib.parse.urlparse` - parsed = urllib.parse.urlparse(url)
- `core/adapters/chrome_cdp_transport.py:127` [adapters] `CdpPolicyError` - raise CdpPolicyError(f"CDP target scheme {parsed.scheme!r} is not a websocket scheme")
- `core/adapters/chrome_cdp_transport.py:201` [adapters] `RuntimeError` - raise RuntimeError("websocket-client is required for Chrome CDP control") from exc
- `core/adapters/chrome_cdp_transport.py:216` [adapters] `websocket.create_connection` - ws = websocket.create_connection(url, timeout=budget)
- `core/adapters/chrome_cdp_transport.py:272` [adapters] `logger.debug` - logger.debug("CDP websocket close failed: %s", exc)
- `core/bus/sensory_gate.py:530` [bus] `urllib.parse.quote` - f"&search={urllib.parse.quote(query)}&limit=3&namespace=0&format=json"
- `core/capabilities/web_interlocutor.py:333` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(text)
- `core/capabilities/web_interlocutor.py:497` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(text)
- `core/capabilities/web_interlocutor.py:563` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(cleaned)
- `core/capabilities/web_interlocutor.py:576` [capabilities] `str` - parts = urllib.parse.urlparse(str(ws_url or ""))
- `core/capabilities/web_interlocutor.py:576` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(str(ws_url or ""))
- `core/capabilities/web_interlocutor.py:866` [capabilities] `urllib.parse.quote` - quoted = urllib.parse.quote(target_url, safe=":/?&=%#")
- `core/capabilities/web_interlocutor.py:4365` [capabilities] `str` - parts = urllib.parse.urlparse(str(url or "").strip())
- `core/capabilities/web_interlocutor.py:4365` [capabilities] `str.strip` - parts = urllib.parse.urlparse(str(url or "").strip())
- `core/capabilities/web_interlocutor.py:4365` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(str(url or "").strip())
- `core/capabilities/web_interlocutor.py:4421` [capabilities] `urllib.parse.urlparse` - current_parts = urllib.parse.urlparse(current)
- `core/capabilities/web_interlocutor.py:4422` [capabilities] `urllib.parse.urlparse` - desired_parts = urllib.parse.urlparse(desired)
- `core/collective/swarm_protocol.py:27` [collective] `socket.gethostname` - self.node_id = socket.gethostname()
- `core/collective/swarm_protocol.py:67` [collective] `logger.warning` - logger.warning("🕸️ Mycelial Swarm running in offline-only mode; socket binding unavailable.")
- `core/collective/swarm_protocol.py:98` [collective] `logger.debug` - logger.debug("Swarm listener close timed out; abandoning socket.")
- `core/communication/messages_history.py:42` [communication] `str` - quoted = urllib.parse.quote(str(self.db_path), safe="/")
- `core/communication/messages_history.py:42` [communication] `urllib.parse.quote` - quoted = urllib.parse.quote(str(self.db_path), safe="/")
- `core/consciousness/heartbeat.py:191` [consciousness] `socket.socket` - with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
- `core/consciousness/heartbeat.py:199` [consciousness] `logger.debug` - logger.debug("Failed to emit keep-alive socket message to watchdog: %s", e)
- `core/conversation/capability_preconditions.py:109` [conversation] `socket.create_connection` - with socket.create_connection((host, port), timeout=_PROBE_TIMEOUT_SECONDS):

## Degradation Handling

- Total `record_degradation()` calls: 4263
- Log-and-limp candidates: 3870
- Nearby fail-closed candidates: 393

Top limp-on files:

- `core/brain/llm/context_assembler.py`: 39
- `core/brain/cognitive_engine.py`: 37
- `core/brain/inference_gate.py`: 35
- `core/consciousness/consciousness_bridge.py`: 29
- `core/senses/voice_engine.py`: 29
- `core/memory/memory_facade.py`: 27
- `core/memory/episodic_memory.py`: 24
- `core/reality_reach/attachments.py`: 24
- `core/resilience/memory_governor.py`: 24
- `core/runtime/runtime_hygiene.py`: 24

## Non-Runtime Candidates

- `core/architect/proof_obligations.py`
- `core/autonomy/autonomous_research_orchestrator.py`
- `core/autonomy/research_cycle.py`
- `core/autonomy/research_goal_filter.py`
- `core/autonomy/research_triggers.py`
- `core/brain/llm/latent_cortex/experiments.py`
- `core/brain/llm/latent_cortex/research_oracle_arbitration.py`
- `core/brain/narrative_memory.py`
- `core/cognition/actr_activation.py`
- `core/consciousness/animal_cognition.py`
- `core/consciousness/narrative_gravity.py`
- `core/consciousness/oscillatory_binding.py`
- `core/evaluation/behavioral_proof.py`
- `core/evaluation/proof_acceptance.py`
- `core/factory/repo_cartographer.py`
- `core/identity/narrative_thread.py`
- `core/lab/experiment_designer.py`
- `core/lab/research_lab.py`
- `core/lab/research_memory.py`
- `core/learning/proof_obligations.py`
- `core/learning/recurrent_sft_sampling.py`
- `core/learning/structured_sft_research_authority.py`
- `core/learning/structured_sft_research_state.py`
- `core/learning/verified_transition_trainer.py`
- `core/learning/verified_transition_update.py`
- `core/memory/hippocampus.py`
- `core/reasoning/proof_answer_domains.py`
- `core/reasoning/proof_answer_solver.py`
- `core/reasoning/proof_answer_types.py`
- `core/reasoning/proof_kernel.py`
- `core/reproducibility/proof_substrate.py`
- `core/runtime/proof_kernel_bridge.py`
- `core/runtime/proof_policy.py`
- `core/search/research_pipeline.py`
- `core/skills/deep_research.py`

## Consolidation Candidates

- `core/audits/`: 2 file(s), 314 line(s)
- `core/coherence/`: 2 file(s), 407 line(s)
- `core/consent/`: 2 file(s), 167 line(s)
- `core/continuity/`: 1 file(s), 33 line(s)
- `core/control/`: 2 file(s), 585 line(s)
- `core/creativity/`: 2 file(s), 801 line(s)
- `core/ethics/`: 2 file(s), 602 line(s)
- `core/evals/`: 2 file(s), 686 line(s)
- `core/initializers/`: 2 file(s), 61 line(s)
- `core/intent/`: 2 file(s), 692 line(s)
- `core/latent/`: 1 file(s), 56 line(s)
- `core/maintenance/`: 2 file(s), 295 line(s)
- `core/middleware/`: 1 file(s), 214 line(s)
- `core/multimodal/`: 2 file(s), 185 line(s)
- `core/networking/`: 1 file(s), 332 line(s)
- `core/ontology/`: 2 file(s), 169 line(s)
- `core/persistence/`: 2 file(s), 619 line(s)
- `core/play/`: 1 file(s), 259 line(s)
- `core/predictive/`: 2 file(s), 186 line(s)
- `core/reproducibility/`: 2 file(s), 497 line(s)
- `core/science/`: 1 file(s), 139 line(s)
- `core/search/`: 2 file(s), 1939 line(s)
- `core/sensors/`: 1 file(s), 195 line(s)
- `core/services/`: 2 file(s), 31 line(s)
- `core/sim/`: 2 file(s), 452 line(s)
- `core/skill_management/`: 1 file(s), 367 line(s)
- `core/transparency/`: 2 file(s), 376 line(s)
- `core/twins/`: 1 file(s), 97 line(s)
