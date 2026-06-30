# Aura Architecture Dependency Map

Schema: `aura.architecture.dependency_map.v2`
Root: `<AURA_ROOT>`
Generated: `0.0`

## Summary

- Subsystems: 146
- Python files: 2050
- Python lines: 576569
- Dependency edges: 869
- ServiceContainer `.get()` calls: 1666
- ServiceContainer registrations: 409
- Boot contract: PASS

## Subsystem Dependency Graph

```mermaid
graph TD
    runtime["runtime<br/>113 files, 27349 lines"]
    utils["utils<br/>44 files, 5993 lines"]
    brain["brain<br/>154 files, 60027 lines"]
    memory["memory<br/>93 files, 21990 lines"]
    consciousness["consciousness<br/>139 files, 65205 lines"]
    resilience["resilience<br/>54 files, 12606 lines"]
    health["health<br/>4 files, 959 lines"]
    agency["agency<br/>43 files, 16301 lines"]
    governance["governance<br/>10 files, 3480 lines"]
    affect["affect<br/>10 files, 3852 lines"]
    identity["identity<br/>17 files, 2354 lines"]
    adaptation["adaptation<br/>27 files, 12670 lines"]
    constitution["constitution<br/>1 files, 25 lines"]
    self_modification["self_modification<br/>32 files, 12043 lines"]
    senses["senses<br/>25 files, 5704 lines"]
    observability["observability<br/>3 files, 575 lines"]
    security["security<br/>33 files, 7450 lines"]
    state["state<br/>6 files, 3716 lines"]
    executive["executive<br/>11 files, 3046 lines"]
    perception["perception<br/>23 files, 6716 lines"]
    orchestrator["orchestrator<br/>43 files, 19557 lines"]
    organism["organism<br/>8 files, 1875 lines"]
    world_model["world_model<br/>11 files, 3278 lines"]
    continuity["continuity<br/>7 files, 238 lines"]
    conversation["conversation<br/>10 files, 7361 lines"]
    skills["skills<br/>82 files, 23481 lines"]
    being["being<br/>25 files, 5782 lines"]
    learning["learning<br/>28 files, 7592 lines"]
    phases["phases<br/>29 files, 19495 lines"]
    reasoning["reasoning<br/>11 files, 4979 lines"]
    social["social<br/>19 files, 5145 lines"]
    tasks["tasks<br/>3 files, 451 lines"]
    world["world<br/>24 files, 1483 lines"]
    actuators["actuators<br/>9 files, 2127 lines"]
    autonomy["autonomy<br/>23 files, 8183 lines"]
    cognition["cognition<br/>16 files, 5391 lines"]
    morality["morality<br/>14 files, 652 lines"]
    self["self<br/>7 files, 2196 lines"]
    autonomic["autonomic<br/>5 files, 1175 lines"]
    capabilities["capabilities<br/>14 files, 6409 lines"]
    coordinators["coordinators<br/>9 files, 4274 lines"]
    ethics["ethics<br/>2 files, 580 lines"]
    kernel["kernel<br/>11 files, 6332 lines"]
    managers["managers<br/>6 files, 955 lines"]
    meta["meta<br/>7 files, 1267 lines"]
    ops["ops<br/>11 files, 2469 lines"]
    phenomenal_substrate["phenomenal_substrate<br/>11 files, 1040 lines"]
    supervisor["supervisor<br/>3 files, 613 lines"]
    unity["unity<br/>11 files, 2733 lines"]
    values["values<br/>11 files, 1120 lines"]
    voice["voice<br/>9 files, 3832 lines"]
    agi["agi<br/>6 files, 1520 lines"]
    bus["bus<br/>5 files, 2321 lines"]
    cognitive["cognitive<br/>12 files, 9212 lines"]
    collective["collective<br/>6 files, 2046 lines"]
    embodiment["embodiment<br/>15 files, 2656 lines"]
    evaluation["evaluation<br/>13 files, 2617 lines"]
    goals["goals<br/>9 files, 3371 lines"]
    motivation["motivation<br/>7 files, 1210 lines"]
    promotion["promotion<br/>6 files, 936 lines"]
    resource["resource<br/>2 files, 430 lines"]
    sandbox["sandbox<br/>4 files, 612 lines"]
    advanced_cognition["advanced_cognition<br/>13 files, 2905 lines"]
    conversational["conversational<br/>4 files, 2239 lines"]
    data["data<br/>3 files, 651 lines"]
    db["db<br/>4 files, 584 lines"]
    morphogenesis["morphogenesis<br/>12 files, 2861 lines"]
    pneuma["pneuma<br/>7 files, 1258 lines"]
    search["search<br/>2 files, 1725 lines"]
    somatic["somatic<br/>5 files, 2383 lines"]
    verification["verification<br/>4 files, 350 lines"]
    workspace["workspace<br/>9 files, 1242 lines"]
    architect["architect<br/>25 files, 5743 lines"]
    coherence["coherence<br/>2 files, 397 lines"]
    discovery["discovery<br/>4 files, 579 lines"]
    environment["environment<br/>82 files, 8528 lines"]
    environments["environments<br/>7 files, 748 lines"]
    evolution["evolution<br/>6 files, 1896 lines"]
    introspection["introspection<br/>3 files, 743 lines"]
    lattice["lattice<br/>5 files, 704 lines"]
    llm["llm<br/>2 files, 19 lines"]
    maintenance["maintenance<br/>2 files, 265 lines"]
    persistence["persistence<br/>2 files, 617 lines"]
    planning["planning<br/>6 files, 2386 lines"]
    predictive["predictive<br/>2 files, 186 lines"]
    self_improvement["self_improvement<br/>12 files, 2388 lines"]
    sensors["sensors<br/>1 files, 159 lines"]
    services["services<br/>2 files, 31 lines"]
    simulation["simulation<br/>3 files, 401 lines"]
    soma["soma<br/>3 files, 513 lines"]
    sovereign["sovereign<br/>4 files, 554 lines"]
    startup["startup<br/>2 files, 330 lines"]
    audit["audit<br/>6 files, 537 lines"]
    body["body<br/>22 files, 1374 lines"]
    consent["consent<br/>2 files, 167 lines"]
    context["context<br/>4 files, 1215 lines"]
    creativity["creativity<br/>2 files, 801 lines"]
    curriculum["curriculum<br/>7 files, 657 lines"]
    cybernetics["cybernetics<br/>6 files, 1133 lines"]
    epistemics["epistemics<br/>7 files, 591 lines"]
    evals["evals<br/>2 files, 280 lines"]
    factory["factory<br/>8 files, 758 lines"]
    grounding["grounding<br/>7 files, 1095 lines"]
    guardians["guardians<br/>7 files, 935 lines"]
    knowledge["knowledge<br/>7 files, 322 lines"]
    media["media<br/>2 files, 273 lines"]
    middleware["middleware<br/>2 files, 254 lines"]
    networking["networking<br/>1 files, 318 lines"]
    plasticity["plasticity<br/>4 files, 342 lines"]
    research_core["research_core<br/>5 files, 580 lines"]
    safety["safety<br/>3 files, 629 lines"]
    sim["sim<br/>7 files, 580 lines"]
    skill_management["skill_management<br/>1 files, 367 lines"]
    sleep["sleep<br/>7 files, 254 lines"]
    sovereignty["sovereignty<br/>3 files, 885 lines"]
    transparency["transparency<br/>2 files, 317 lines"]
    twins["twins<br/>1 files, 97 lines"]
    unknowns["unknowns<br/>4 files, 325 lines"]
    welfare["welfare<br/>7 files, 228 lines"]
    actuation["actuation<br/>9 files, 350 lines"]
    adapters["adapters<br/>3 files, 402 lines"]
    audits["audits<br/>2 files, 267 lines"]
    control["control<br/>2 files, 586 lines"]
    core_root["core_root<br/>177 files, 56543 lines"]
    council["council<br/>5 files, 533 lines"]
    distributed["distributed<br/>3 files, 140 lines"]
    forge["forge<br/>8 files, 325 lines"]
    initializers["initializers<br/>2 files, 140 lines"]
    intent["intent<br/>1 files, 68 lines"]
    lab["lab<br/>7 files, 378 lines"]
    latent["latent<br/>1 files, 56 lines"]
    mission["mission<br/>4 files, 472 lines"]
    multimodal["multimodal<br/>2 files, 185 lines"]
    neuroweb["neuroweb<br/>5 files, 368 lines"]
    ontology["ontology<br/>2 files, 169 lines"]
    pipeline["pipeline<br/>3 files, 217 lines"]
    play["play<br/>1 files, 228 lines"]
    providers["providers<br/>6 files, 1300 lines"]
    reproducibility["reproducibility<br/>2 files, 497 lines"]
    science["science<br/>1 files, 139 lines"]
    session["session<br/>2 files, 231 lines"]
    swarm["swarm<br/>5 files, 396 lines"]
    systems["systems<br/>3 files, 256 lines"]
    telemetry["telemetry<br/>2 files, 191 lines"]
    temporal["temporal<br/>3 files, 1507 lines"]
    tools["tools<br/>9 files, 869 lines"]
    runtime --> actuators
    runtime --> adaptation
    runtime --> affect
    runtime --> agency
    runtime --> architect
    runtime --> autonomy
    runtime --> being
    runtime --> brain
    runtime --> consciousness
    runtime --> constitution
    runtime --> conversation
    runtime --> evaluation
    runtime --> governance
    runtime --> health
    runtime --> identity
    runtime --> memory
    runtime --> observability
    runtime --> organism
    runtime --> perception
    runtime --> persistence
    runtime --> phases
    runtime --> research_core
    runtime --> resilience
    runtime --> security
    runtime --> self
    runtime --> self_modification
    runtime --> social
    runtime --> state
    runtime --> supervisor
    runtime --> utils
    runtime --> workspace
    utils --> consciousness
    utils --> health
    utils --> identity
    utils --> managers
    utils --> memory
    utils --> observability
    utils --> resilience
    utils --> runtime
    utils --> tasks
    brain --> adaptation
    brain --> affect
    brain --> agency
    brain --> agi
    brain --> being
    brain --> cognitive
    brain --> consciousness
    brain --> constitution
    brain --> continuity
    brain --> conversation
    brain --> health
    brain --> identity
    brain --> kernel
    brain --> memory
    brain --> morphogenesis
    brain --> observability
    brain --> ops
    brain --> organism
    brain --> phases
    brain --> pneuma
    brain --> reasoning
    brain --> resilience
    brain --> runtime
    brain --> search
    brain --> security
    brain --> self
    brain --> senses
    brain --> state
    brain --> utils
    brain --> voice
    memory --> actuators
    memory --> being
    memory --> brain
    memory --> consciousness
    memory --> constitution
    memory --> db
    memory --> governance
    memory --> health
    memory --> observability
    memory --> phases
    memory --> resilience
    memory --> runtime
    memory --> utils
    memory --> values
    consciousness --> actuators
    consciousness --> adaptation
    consciousness --> affect
    consciousness --> agency
    consciousness --> being
    consciousness --> brain
    consciousness --> constitution
    consciousness --> coordinators
    consciousness --> evaluation
    consciousness --> goals
    consciousness --> health
    consciousness --> memory
    consciousness --> meta
    consciousness --> observability
    consciousness --> orchestrator
    consciousness --> phenomenal_substrate
    consciousness --> pneuma
    consciousness --> predictive
    consciousness --> reasoning
    consciousness --> resilience
    consciousness --> runtime
    consciousness --> senses
    consciousness --> sensors
    consciousness --> state
    consciousness --> unity
    consciousness --> utils
    consciousness --> world
    consciousness --> world_model
    resilience --> agency
    resilience --> brain
    resilience --> consciousness
    resilience --> conversation
    resilience --> coordinators
    resilience --> health
    resilience --> memory
    resilience --> runtime
    resilience --> tasks
    resilience --> utils
    health --> brain
    health --> memory
    health --> runtime
    health --> state
    health --> utils
    agency --> adaptation
    agency --> affect
    agency --> agi
    agency --> brain
    agency --> cognition
    agency --> consciousness
    agency --> constitution
    agency --> continuity
    agency --> governance
    agency --> health
    agency --> identity
    agency --> orchestrator
    agency --> organism
    agency --> resilience
    agency --> runtime
    agency --> skills
    agency --> social
    agency --> tasks
    agency --> utils
    agency --> values
    governance --> actuators
    governance --> being
    governance --> brain
    governance --> consciousness
    governance --> identity
    governance --> memory
    governance --> runtime
    governance --> utils
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
    identity --> agency
    identity --> brain
    identity --> governance
    identity --> organism
    identity --> runtime
    identity --> utils
    adaptation --> actuators
    adaptation --> affect
    adaptation --> brain
    adaptation --> cognitive
    adaptation --> executive
    adaptation --> health
    adaptation --> identity
    adaptation --> learning
    adaptation --> memory
    adaptation --> resilience
    adaptation --> runtime
    adaptation --> sensors
    adaptation --> utils
    adaptation --> world
    self_modification --> ethics
    self_modification --> governance
    self_modification --> memory
    self_modification --> resilience
    self_modification --> runtime
    self_modification --> skills
    self_modification --> utils
    senses --> affect
    senses --> brain
    senses --> consciousness
    senses --> constitution
    senses --> health
    senses --> networking
    senses --> orchestrator
    senses --> perception
    senses --> resilience
    senses --> runtime
    senses --> security
    senses --> supervisor
    senses --> utils
    observability --> runtime
    security --> affect
    security --> agency
    security --> brain
    security --> consciousness
    security --> identity
    security --> memory
    security --> perception
    security --> runtime
    security --> utils
    state --> bus
    state --> constitution
    state --> governance
    state --> memory
    state --> motivation
    state --> runtime
    state --> unity
    state --> utils
    executive --> agency
    executive --> autonomy
    executive --> consciousness
    executive --> constitution
    executive --> continuity
    executive --> goals
    executive --> governance
    executive --> health
    executive --> memory
    executive --> morality
    executive --> organism
    executive --> runtime
    executive --> state
    executive --> utils
    perception --> brain
    perception --> capabilities
    perception --> phenomenal_substrate
    perception --> resilience
    perception --> runtime
    perception --> security
    perception --> utils
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
    orchestrator --> cognitive
    orchestrator --> collective
    orchestrator --> consciousness
    orchestrator --> constitution
    orchestrator --> continuity
    orchestrator --> conversation
    orchestrator --> coordinators
    orchestrator --> data
    orchestrator --> db
    orchestrator --> embodiment
    orchestrator --> environment
    orchestrator --> ethics
    orchestrator --> evals
    orchestrator --> evolution
    orchestrator --> executive
    orchestrator --> goals
    orchestrator --> governance
    orchestrator --> guardians
    orchestrator --> health
    orchestrator --> identity
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
    orchestrator --> pneuma
    orchestrator --> resilience
    orchestrator --> runtime
    orchestrator --> safety
    orchestrator --> security
    orchestrator --> self
    orchestrator --> self_improvement
    orchestrator --> self_modification
    orchestrator --> senses
    orchestrator --> sim
    orchestrator --> simulation
    orchestrator --> skill_management
    orchestrator --> soma
    orchestrator --> sovereignty
    orchestrator --> startup
    orchestrator --> state
    orchestrator --> supervisor
    orchestrator --> tasks
    orchestrator --> utils
    orchestrator --> verification
    orchestrator --> voice
    orchestrator --> world_model
    organism --> agency
    organism --> body
    organism --> executive
    organism --> health
    organism --> identity
    organism --> memory
    organism --> resilience
    organism --> runtime
    organism --> sleep
    organism --> utils
    organism --> values
    organism --> welfare
    organism --> workspace
    organism --> world
    world_model --> advanced_cognition
    world_model --> brain
    world_model --> cognition
    world_model --> constitution
    world_model --> health
    world_model --> resilience
    world_model --> runtime
    world_model --> values
    continuity --> identity
    continuity --> organism
    continuity --> runtime
    conversation --> brain
    conversation --> consciousness
    conversation --> memory
    conversation --> organism
    conversation --> runtime
    conversation --> social
    conversation --> utils
    skills --> actuators
    skills --> advanced_cognition
    skills --> affect
    skills --> being
    skills --> brain
    skills --> capabilities
    skills --> consent
    skills --> conversation
    skills --> embodiment
    skills --> executive
    skills --> governance
    skills --> learning
    skills --> memory
    skills --> perception
    skills --> runtime
    skills --> sandbox
    skills --> search
    skills --> security
    skills --> self_modification
    skills --> senses
    skills --> sovereign
    skills --> utils
    being --> governance
    being --> runtime
    learning --> brain
    learning --> consciousness
    learning --> introspection
    learning --> memory
    learning --> promotion
    learning --> reasoning
    learning --> runtime
    learning --> self_modification
    learning --> skills
    learning --> tasks
    learning --> utils
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
    phases --> memory
    phases --> morality
    phases --> reasoning
    phases --> runtime
    phases --> self_modification
    phases --> skills
    phases --> social
    phases --> somatic
    phases --> state
    phases --> unity
    phases --> utils
    phases --> voice
    reasoning --> observability
    reasoning --> runtime
    social --> agency
    social --> ethics
    social --> governance
    social --> runtime
    social --> utils
    tasks --> runtime
    world --> governance
    world --> runtime
    actuators --> affect
    actuators --> brain
    actuators --> executive
    actuators --> memory
    actuators --> runtime
    actuators --> sandbox
    actuators --> search
    actuators --> skills
    actuators --> utils
    actuators --> world
    autonomy --> affect
    autonomy --> agency
    autonomy --> consciousness
    autonomy --> executive
    autonomy --> governance
    autonomy --> memory
    autonomy --> observability
    autonomy --> resource
    autonomy --> runtime
    autonomy --> state
    autonomy --> utils
    cognition --> consciousness
    cognition --> governance
    cognition --> memory
    cognition --> runtime
    cognition --> social
    cognition --> world_model
    morality --> brain
    morality --> runtime
    morality --> utils
    self --> affect
    self --> bus
    self --> consciousness
    self --> memory
    self --> runtime
    self --> security
    self --> senses
    self --> state
    self --> utils
    autonomic --> orchestrator
    autonomic --> runtime
    autonomic --> utils
    capabilities --> memory
    capabilities --> perception
    capabilities --> phenomenal_substrate
    capabilities --> planning
    capabilities --> runtime
    capabilities --> security
    capabilities --> self
    capabilities --> skills
    capabilities --> voice
    coordinators --> autonomy
    coordinators --> brain
    coordinators --> continuity
    coordinators --> evolution
    coordinators --> health
    coordinators --> maintenance
    coordinators --> memory
    coordinators --> meta
    coordinators --> morphogenesis
    coordinators --> observability
    coordinators --> ops
    coordinators --> orchestrator
    coordinators --> persistence
    coordinators --> resilience
    coordinators --> resource
    coordinators --> runtime
    coordinators --> security
    coordinators --> somatic
    coordinators --> tasks
    coordinators --> utils
    coordinators --> world_model
    ethics --> brain
    ethics --> morality
    ethics --> runtime
    ethics --> utils
    kernel --> agency
    kernel --> brain
    kernel --> cognition
    kernel --> consciousness
    kernel --> continuity
    kernel --> cybernetics
    kernel --> executive
    kernel --> health
    kernel --> learning
    kernel --> orchestrator
    kernel --> perception
    kernel --> phases
    kernel --> resilience
    kernel --> runtime
    kernel --> security
    kernel --> self_modification
    kernel --> senses
    kernel --> somatic
    kernel --> state
    kernel --> utils
    managers --> autonomic
    managers --> brain
    managers --> collective
    managers --> constitution
    managers --> data
    managers --> health
    managers --> memory
    managers --> ops
    managers --> orchestrator
    managers --> resilience
    managers --> runtime
    managers --> security
    managers --> self_modification
    managers --> senses
    managers --> utils
    meta --> adaptation
    meta --> runtime
    meta --> utils
    ops --> brain
    ops --> kernel
    ops --> managers
    ops --> observability
    ops --> resilience
    ops --> resource
    ops --> runtime
    ops --> senses
    ops --> state
    ops --> supervisor
    ops --> utils
    phenomenal_substrate --> runtime
    supervisor --> bus
    supervisor --> runtime
    unity --> affect
    unity --> cognition
    unity --> consciousness
    unity --> runtime
    unity --> social
    unity --> values
    values --> agency
    values --> governance
    values --> runtime
    values --> social
    voice --> brain
    voice --> conversational
    voice --> executive
    voice --> resilience
    voice --> runtime
    voice --> senses
    voice --> utils
    agi --> adaptation
    agi --> brain
    agi --> constitution
    agi --> embodiment
    agi --> executive
    agi --> grounding
    agi --> health
    agi --> runtime
    agi --> utils
    agi --> world_model
    bus --> resilience
    bus --> runtime
    bus --> utils
    cognitive --> brain
    cognitive --> health
    cognitive --> phases
    cognitive --> runtime
    cognitive --> utils
    collective --> adaptation
    collective --> agency
    collective --> brain
    collective --> runtime
    collective --> utils
    embodiment --> agency
    embodiment --> consciousness
    embodiment --> environments
    embodiment --> ethics
    embodiment --> governance
    embodiment --> organism
    embodiment --> runtime
    embodiment --> utils
    evaluation --> conversation
    evaluation --> learning
    evaluation --> promotion
    evaluation --> runtime
    goals --> agency
    goals --> brain
    goals --> runtime
    goals --> state
    goals --> utils
    motivation --> brain
    motivation --> consciousness
    motivation --> constitution
    motivation --> health
    motivation --> runtime
    motivation --> utils
    promotion --> runtime
    resource --> observability
    resource --> resilience
    resource --> runtime
    sandbox --> runtime
    advanced_cognition --> reasoning
    advanced_cognition --> runtime
    conversational --> memory
    conversational --> runtime
    data --> runtime
    db --> runtime
    morphogenesis --> adaptation
    morphogenesis --> memory
    morphogenesis --> resilience
    morphogenesis --> runtime
    morphogenesis --> self_modification
    pneuma --> affect
    pneuma --> runtime
    pneuma --> utils
    search --> runtime
    search --> utils
    somatic --> memory
    somatic --> runtime
    somatic --> utils
    verification --> discovery
    verification --> middleware
    workspace --> runtime
    architect --> adaptation
    architect --> brain
    architect --> consciousness
    architect --> runtime
    architect --> self_modification
    architect --> world_model
    coherence --> agency
    coherence --> consciousness
    coherence --> runtime
    coherence --> self
    coherence --> unity
    discovery --> runtime
    discovery --> self_modification
    environment --> advanced_cognition
    environment --> brain
    environment --> consciousness
    environment --> environments
    environment --> executive
    environment --> memory
    environment --> perception
    environment --> runtime
    environments --> environment
    environments --> perception
    environments --> runtime
    evolution --> agi
    evolution --> brain
    evolution --> runtime
    evolution --> utils
    introspection --> runtime
    llm --> brain
    maintenance --> resilience
    maintenance --> runtime
    persistence --> observability
    persistence --> resilience
    persistence --> runtime
    planning --> brain
    planning --> capabilities
    planning --> runtime
    predictive --> brain
    predictive --> runtime
    predictive --> utils
    self_improvement --> brain
    self_improvement --> llm
    self_improvement --> runtime
    self_improvement --> self_modification
    sensors --> world
    services --> autonomic
    simulation --> brain
    simulation --> consciousness
    simulation --> identity
    simulation --> runtime
    simulation --> world_model
    soma --> resilience
    soma --> runtime
    soma --> utils
    sovereign --> runtime
    startup --> brain
    startup --> runtime
    startup --> senses
    audit --> epistemics
    audit --> runtime
    body --> capabilities
    body --> perception
    body --> runtime
    body --> security
    context --> runtime
    creativity --> memory
    creativity --> runtime
    curriculum --> runtime
    cybernetics --> cognitive
    cybernetics --> kernel
    cybernetics --> runtime
    cybernetics --> utils
    factory --> runtime
    grounding --> plasticity
    grounding --> runtime
    guardians --> brain
    guardians --> morality
    guardians --> runtime
    guardians --> tasks
    guardians --> utils
    knowledge --> brain
    knowledge --> runtime
    knowledge --> utils
    middleware --> runtime
    networking --> runtime
    plasticity --> runtime
    research_core --> curriculum
    research_core --> discovery
    research_core --> lattice
    research_core --> promotion
    research_core --> runtime
    research_core --> unknowns
    research_core --> verification
    safety --> runtime
    sim --> brain
    sim --> morality
    sim --> twins
    sim --> utils
    skill_management --> resilience
    skill_management --> runtime
    skill_management --> self_modification
    sleep --> identity
    sleep --> memory
    sleep --> runtime
    sovereignty --> ethics
    sovereignty --> governance
    sovereignty --> identity
    sovereignty --> organism
    sovereignty --> runtime
    sovereignty --> utils
    transparency --> runtime
    unknowns --> lattice
    unknowns --> promotion
    unknowns --> verification
    actuation --> runtime
    adapters --> runtime
    audits --> brain
    audits --> runtime
    control --> runtime
    control --> utils
    core_root --> adaptation
    core_root --> affect
    core_root --> agency
    core_root --> architect
    core_root --> autonomic
    core_root --> autonomy
    core_root --> being
    core_root --> brain
    core_root --> coherence
    core_root --> collective
    core_root --> consciousness
    core_root --> constitution
    core_root --> context
    core_root --> continuity
    core_root --> conversation
    core_root --> conversational
    core_root --> coordinators
    core_root --> data
    core_root --> evaluation
    core_root --> executive
    core_root --> goals
    core_root --> governance
    core_root --> health
    core_root --> identity
    core_root --> llm
    core_root --> managers
    core_root --> media
    core_root --> memory
    core_root --> meta
    core_root --> motivation
    core_root --> observability
    core_root --> orchestrator
    core_root --> phases
    core_root --> planning
    core_root --> predictive
    core_root --> reasoning
    core_root --> resilience
    core_root --> resource
    core_root --> runtime
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
    core_root --> world_model
    council --> runtime
    forge --> runtime
    initializers --> adaptation
    initializers --> consciousness
    initializers --> introspection
    initializers --> memory
    initializers --> meta
    initializers --> runtime
    initializers --> senses
    initializers --> utils
    intent --> runtime
    mission --> runtime
    multimodal --> runtime
    neuroweb --> brain
    neuroweb --> consciousness
    neuroweb --> runtime
    pipeline --> runtime
    play --> consciousness
    play --> runtime
    providers --> affect
    providers --> brain
    providers --> cognition
    providers --> collective
    providers --> consciousness
    providers --> continuity
    providers --> coordinators
    providers --> creativity
    providers --> db
    providers --> learning
    providers --> managers
    providers --> memory
    providers --> motivation
    providers --> ops
    providers --> orchestrator
    providers --> perception
    providers --> phenomenal_substrate
    providers --> reasoning
    providers --> resilience
    providers --> runtime
    providers --> self_modification
    providers --> senses
    providers --> services
    providers --> unity
    providers --> utils
    providers --> world_model
    reproducibility --> runtime
    science --> runtime
    science --> world
    session --> runtime
    swarm --> factory
    swarm --> runtime
    swarm --> sandbox
    swarm --> world
    systems --> runtime
    systems --> services
    temporal --> runtime
    temporal --> utils
    tools --> runtime
    tools --> sandbox
    tools --> skills
```

## Core Subsystem Stats

| Subsystem | Files | Lines | Bytes | Deps Out | Deps In |
| --- | ---: | ---: | ---: | ---: | ---: |
| consciousness | 139 | 65205 | 2758794 | 41 | 31 |
| brain | 154 | 60027 | 2576232 | 43 | 48 |
| core_root | 177 | 56543 | 2309356 | 100 | 0 |
| runtime | 113 | 27349 | 979924 | 46 | 126 |
| skills | 82 | 23481 | 976209 | 32 | 8 |
| memory | 93 | 21990 | 893773 | 21 | 35 |
| orchestrator | 43 | 19557 | 865244 | 132 | 9 |
| phases | 29 | 19495 | 885064 | 35 | 7 |
| agency | 43 | 16301 | 663081 | 33 | 19 |
| adaptation | 27 | 12670 | 506593 | 22 | 14 |
| resilience | 54 | 12606 | 509722 | 17 | 27 |
| self_modification | 32 | 12043 | 476186 | 13 | 14 |
| cognitive | 12 | 9212 | 373941 | 9 | 4 |
| environment | 82 | 8528 | 332940 | 11 | 2 |
| autonomy | 23 | 8183 | 335291 | 18 | 6 |
| learning | 28 | 7592 | 303464 | 16 | 7 |
| security | 33 | 7450 | 301098 | 15 | 13 |
| conversation | 10 | 7361 | 273639 | 12 | 8 |
| perception | 23 | 6716 | 270387 | 12 | 11 |
| capabilities | 14 | 6409 | 246987 | 11 | 5 |
| kernel | 11 | 6332 | 264377 | 25 | 5 |
| utils | 44 | 5993 | 232640 | 18 | 59 |
| being | 25 | 5782 | 225115 | 4 | 7 |
| architect | 25 | 5743 | 240043 | 10 | 2 |
| senses | 25 | 5704 | 237906 | 20 | 14 |
| cognition | 16 | 5391 | 226608 | 11 | 6 |
| social | 19 | 5145 | 221975 | 9 | 7 |
| reasoning | 11 | 4979 | 199749 | 3 | 7 |
| coordinators | 9 | 4274 | 200629 | 35 | 5 |
| affect | 10 | 3852 | 176223 | 14 | 16 |
| voice | 9 | 3832 | 169898 | 11 | 5 |
| state | 6 | 3716 | 158306 | 10 | 13 |
| governance | 10 | 3480 | 143151 | 14 | 18 |
| goals | 9 | 3371 | 144025 | 10 | 4 |
| world_model | 11 | 3278 | 138218 | 12 | 9 |
| executive | 11 | 3046 | 126322 | 17 | 11 |
| advanced_cognition | 13 | 2905 | 118305 | 3 | 3 |
| morphogenesis | 12 | 2861 | 110971 | 7 | 3 |
| unity | 11 | 2733 | 114640 | 7 | 5 |
| embodiment | 15 | 2656 | 103934 | 12 | 4 |
| evaluation | 13 | 2617 | 93160 | 4 | 4 |
| ops | 11 | 2469 | 96268 | 15 | 5 |
| self_improvement | 12 | 2388 | 91090 | 4 | 2 |
| planning | 6 | 2386 | 95080 | 4 | 2 |
| somatic | 5 | 2383 | 90349 | 8 | 3 |
| identity | 17 | 2354 | 98430 | 9 | 15 |
| bus | 5 | 2321 | 95750 | 6 | 4 |
| conversational | 4 | 2239 | 95619 | 4 | 3 |
| self | 7 | 2196 | 91270 | 12 | 6 |
| actuators | 9 | 2127 | 83454 | 13 | 6 |
| collective | 6 | 2046 | 83983 | 8 | 4 |
| evolution | 6 | 1896 | 77054 | 8 | 2 |
| organism | 8 | 1875 | 70075 | 17 | 9 |
| search | 2 | 1725 | 65511 | 7 | 3 |
| agi | 6 | 1520 | 63529 | 13 | 4 |
| temporal | 3 | 1507 | 50941 | 2 | 0 |
| world | 24 | 1483 | 54104 | 3 | 7 |
| body | 22 | 1374 | 48751 | 6 | 1 |
| providers | 6 | 1300 | 57853 | 54 | 0 |
| meta | 7 | 1267 | 47564 | 5 | 5 |
| pneuma | 7 | 1258 | 47673 | 4 | 3 |
| workspace | 9 | 1242 | 45306 | 3 | 3 |
| context | 4 | 1215 | 47010 | 1 | 1 |
| motivation | 7 | 1210 | 51052 | 10 | 4 |
| autonomic | 5 | 1175 | 48010 | 7 | 5 |
| cybernetics | 6 | 1133 | 45264 | 6 | 1 |
| values | 11 | 1120 | 47710 | 5 | 5 |
| grounding | 7 | 1095 | 40528 | 4 | 1 |
| phenomenal_substrate | 11 | 1040 | 41792 | 2 | 5 |
| health | 4 | 959 | 36563 | 8 | 22 |
| managers | 6 | 955 | 40672 | 25 | 5 |
| promotion | 6 | 936 | 31616 | 1 | 4 |
| guardians | 7 | 935 | 39345 | 8 | 1 |
| sovereignty | 3 | 885 | 33782 | 10 | 1 |
| tools | 9 | 869 | 32298 | 4 | 0 |
| creativity | 2 | 801 | 33361 | 3 | 1 |
| factory | 8 | 758 | 29090 | 3 | 1 |
| environments | 7 | 748 | 31101 | 3 | 2 |
| introspection | 3 | 743 | 28738 | 1 | 2 |
| lattice | 5 | 704 | 26089 | 0 | 2 |
| curriculum | 7 | 657 | 21995 | 1 | 1 |
| morality | 14 | 652 | 25702 | 5 | 6 |
| data | 3 | 651 | 22377 | 2 | 3 |
| safety | 3 | 629 | 25738 | 3 | 1 |
| persistence | 2 | 617 | 24953 | 3 | 2 |
| supervisor | 3 | 613 | 23060 | 2 | 5 |
| sandbox | 4 | 612 | 21716 | 1 | 4 |
| epistemics | 7 | 591 | 22459 | 0 | 1 |
| control | 2 | 586 | 21056 | 4 | 0 |
| db | 4 | 584 | 22537 | 2 | 3 |
| ethics | 2 | 580 | 23482 | 6 | 5 |
| research_core | 5 | 580 | 22543 | 8 | 1 |
| sim | 7 | 580 | 21361 | 6 | 1 |
| discovery | 4 | 579 | 20581 | 2 | 2 |
| observability | 3 | 575 | 20634 | 4 | 13 |
| sovereign | 4 | 554 | 19698 | 2 | 2 |
| audit | 6 | 537 | 20846 | 4 | 1 |
| council | 5 | 533 | 21560 | 4 | 0 |
| soma | 3 | 513 | 19932 | 3 | 2 |
| reproducibility | 2 | 497 | 18141 | 1 | 0 |
| mission | 4 | 472 | 17806 | 1 | 0 |
| tasks | 3 | 451 | 16077 | 3 | 7 |
| resource | 2 | 430 | 15691 | 3 | 4 |
| adapters | 3 | 402 | 13469 | 1 | 0 |
| simulation | 3 | 401 | 15916 | 7 | 2 |
| coherence | 2 | 397 | 18920 | 6 | 2 |
| swarm | 5 | 396 | 15170 | 6 | 0 |
| lab | 7 | 378 | 13494 | 0 | 0 |
| neuroweb | 5 | 368 | 14195 | 5 | 0 |
| skill_management | 1 | 367 | 17964 | 6 | 1 |
| actuation | 9 | 350 | 11972 | 2 | 0 |
| verification | 4 | 350 | 13177 | 2 | 3 |
| plasticity | 4 | 342 | 12056 | 2 | 1 |
| startup | 2 | 330 | 11428 | 5 | 2 |
| forge | 8 | 325 | 11877 | 2 | 0 |
| unknowns | 4 | 325 | 11829 | 3 | 1 |
| knowledge | 7 | 322 | 10906 | 5 | 1 |
| networking | 1 | 318 | 11897 | 2 | 1 |
| transparency | 2 | 317 | 12256 | 1 | 1 |
| evals | 2 | 280 | 9864 | 2 | 1 |
| media | 2 | 273 | 9349 | 0 | 1 |
| audits | 2 | 267 | 9492 | 3 | 0 |
| maintenance | 2 | 265 | 9570 | 4 | 2 |
| systems | 3 | 256 | 9861 | 3 | 0 |
| middleware | 2 | 254 | 11019 | 2 | 1 |
| sleep | 7 | 254 | 9719 | 3 | 1 |
| continuity | 7 | 238 | 8314 | 4 | 8 |
| session | 2 | 231 | 9389 | 1 | 0 |
| play | 1 | 228 | 8774 | 4 | 0 |
| welfare | 7 | 228 | 8034 | 0 | 1 |
| pipeline | 3 | 217 | 6684 | 1 | 0 |
| telemetry | 2 | 191 | 5594 | 0 | 0 |
| predictive | 2 | 186 | 7105 | 5 | 2 |
| multimodal | 2 | 185 | 6591 | 1 | 0 |
| ontology | 2 | 169 | 5381 | 0 | 0 |
| consent | 2 | 167 | 5514 | 0 | 1 |
| sensors | 1 | 159 | 5926 | 1 | 2 |
| distributed | 3 | 140 | 4655 | 0 | 0 |
| initializers | 2 | 140 | 6565 | 10 | 0 |
| science | 1 | 139 | 5947 | 4 | 0 |
| twins | 1 | 97 | 3626 | 0 | 1 |
| intent | 1 | 68 | 2661 | 1 | 0 |
| latent | 1 | 56 | 2337 | 0 | 0 |
| services | 2 | 31 | 1171 | 1 | 2 |
| constitution | 1 | 25 | 795 | 0 | 14 |
| llm | 2 | 19 | 745 | 1 | 2 |

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

## ServiceContainer Cross-Wiring

- Unique services retrieved: 411
- Unique services registered: 335
- Services retrieved without detected registration: 198

### Top Fetched Services

| Service | Gets | Registrations |
| --- | ---: | ---: |
| orchestrator | 75 | 3 |
| cognitive_engine | 54 | 3 |
| llm_router | 50 | 2 |
| inference_gate | 47 | 4 |
| affect_engine | 42 | 1 |
| memory_facade | 33 | 1 |
| capability_engine | 33 | 2 |
| liquid_substrate | 29 | 1 |
| mycelial_network | 27 | 2 |
| conscious_substrate | 26 | 2 |
| drive_engine | 25 | 1 |
| free_energy_engine | 25 | 0 |
| homeostasis | 22 | 1 |
| global_workspace | 22 | 2 |
| world_state | 22 | 1 |
| goal_engine | 21 | 0 |
| state_repository | 19 | 1 |
| episodic_memory | 19 | 1 |
| knowledge_graph | 18 | 0 |
| qualia_synthesizer | 18 | 3 |

### Missing Registration Candidates

- `actuator_registry` fetched 1 time(s)
- `adaptive_immune_system` fetched 3 time(s)
- `affect` fetched 2 time(s)
- `affect_engine_v2` fetched 2 time(s)
- `affect_module` fetched 2 time(s)
- `affordance_kb` fetched 1 time(s)
- `agency` fetched 2 time(s)
- `agent_delegator` fetched 6 time(s)
- `alife_dynamics` fetched 1 time(s)
- `alife_extensions` fetched 1 time(s)
- `api_adapter` fetched 5 time(s)
- `archive_engine` fetched 3 time(s)
- `audit_log` fetched 1 time(s)
- `audit_suite` fetched 1 time(s)
- `aura_state` fetched 4 time(s)
- `autonomous_loop` fetched 1 time(s)
- `autonomous_resilience_mesh` fetched 1 time(s)
- `autopoiesis` fetched 1 time(s)
- `belief_challenger` fetched 2 time(s)
- `belief_engine` fetched 1 time(s)
- `belief_system` fetched 1 time(s)
- `bicameral_advisory` fetched 1 time(s)
- `binding_engine` fetched 2 time(s)
- `black_hole_vault` fetched 1 time(s)
- `blackhole_vault` fetched 1 time(s)
- `brain` fetched 5 time(s)
- `brainstem_client` fetched 1 time(s)
- `bryan_model` fetched 3 time(s)
- `canonical_self_engine` fetched 4 time(s)
- `capability_map` fetched 1 time(s)
- `cel_bridge` fetched 2 time(s)
- `cellular_substrate` fetched 1 time(s)
- `cloud_body` fetched 1 time(s)
- `code_refiner` fetched 1 time(s)
- `code_repair` fetched 1 time(s)
- `cognitive_integration_layer` fetched 1 time(s)
- `cognitive_kernel` fetched 3 time(s)
- `cognitive_situation` fetched 1 time(s)
- `coherence_report` fetched 1 time(s)
- `cold_store` fetched 1 time(s)
- `concept_linker` fetched 1 time(s)
- `config` fetched 1 time(s)
- `consciousness_evidence` fetched 1 time(s)
- `constitution` fetched 1 time(s)
- `constitutive_expression_layer` fetched 2 time(s)
- `context_pruner` fetched 1 time(s)
- `continuity` fetched 1 time(s)
- `continuous_experience_stream` fetched 1 time(s)
- `continuous_substrate` fetched 4 time(s)
- `conversation_engine` fetched 1 time(s)

## Operational Authority Map

| Surface | Calls | Files | Owner Calls | Review Candidates |
| --- | ---: | ---: | ---: | ---: |
| UnifiedWill decisions | 60 | 32 | 2 | 58 |
| Memory writes | 319 | 130 | 50 | 269 |
| State mutation | 423 | 160 | 8 | 415 |
| Tool execution | 96 | 49 | 6 | 90 |
| Self-modification and patching | 14 | 12 | 1 | 13 |
| LLM inference | 274 | 165 | 73 | 201 |
| External I/O | 96 | 40 | 8 | 88 |

### UnifiedWill decisions

Calls that can ask the single will authority to approve action.

Review candidates:
- `core/actuators/actuator_synthesis.py:183` [actuators] `get_will` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:183` [actuators] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/adaptive_immunity.py:2061` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/adaptive_immunity.py:2061` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:624` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:624` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:188` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:188` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/agency/agency_bus.py:83` [agency] `get_will` - _auto_decision = get_will().decide(
- `core/agency/agency_bus.py:83` [agency] `get_will.decide` - _auto_decision = get_will().decide(
- `core/agency/hierarchical_agency.py:400` [agency] `get_will` - decision = get_will().decide(
- `core/agency/hierarchical_agency.py:400` [agency] `get_will.decide` - decision = get_will().decide(
- `core/autonomy/genuine_refusal.py:303` [autonomy] `will.decide` - decision = will.decide(content, source="genuine_refusal", domain=domain, priority=0.8, context=ctx)
- `core/autonomy/self_modification.py:315` [autonomy] `will.decide` - decision = will.decide(
- `core/cognitive/autopoiesis.py:961` [cognitive] `will.decide` - decision = will.decide(
- `core/consciousness/parallel_branches.py:736` [consciousness] `will.decide` - decision = will.decide(
- `core/environment/governance_bridge.py:47` [environment] `self.will_gateway.decide` - will_decision = await self.will_gateway.decide(intent)
- `core/goals/goal_engine.py:1020` [goals] `will.decide` - decision = will.decide(
- `core/governance/will_gate.py:109` [governance] `will.decide` - decision = will.decide(
- `core/governance/will_gate.py:160` [governance] `will.decide` - decision = will.decide(
- `core/initiative_synthesis.py:785` [core_root] `get_will` - decision = get_will().decide(
- `core/initiative_synthesis.py:785` [core_root] `get_will.decide` - decision = get_will().decide(
- `core/learning/genuine_learning_pipeline.py:649` [learning] `get_will` - decision = get_will().decide(
- `core/learning/genuine_learning_pipeline.py:649` [learning] `get_will.decide` - decision = get_will().decide(
- `core/learning/recursive_self_improvement.py:529` [learning] `get_will` - decision = get_will().decide(

### Memory writes

Calls that can create durable or semantically promoted memory.

Review candidates:
- `core/actuators/doc_ingest.py:81` [actuators] `get_memory_write_gateway` - gateway = get_memory_write_gateway()
- `core/actuators/doc_ingest.py:95` [actuators] `MemoryWriteRequest` - MemoryWriteRequest(
- `core/actuators/doc_ingest.py:106` [actuators] `memory_facade.add_memory` - maybe_result = memory_facade.add_memory(text=text, metadata=metadata)
- `core/adaptation/abstraction_engine.py:121` [adaptation] `MemoryWriteReceipt` - MemoryWriteReceipt(
- `core/adaptation/abstraction_engine.py:140` [adaptation] `memory_facade.store` - await memory_facade.store(
- `core/adaptation/adaptive_immunity.py:1263` [adaptation] `self._cells.append` - self._cells.append(memory)
- `core/advanced_cognition/continual_learning_stability.py:94` [advanced_cognition] `self._persist_memory` - self._persist_memory(rec)
- `core/advanced_cognition/continual_learning_stability.py:98` [advanced_cognition] `self.store_memory` - return self.store_memory(
- `core/advanced_cognition/continual_learning_stability.py:208` [advanced_cognition] `scored.append` - scored.append((score, memory))
- `core/advanced_cognition/continual_learning_stability.py:313` [advanced_cognition] `self._append_jsonl` - self._append_jsonl(self.state_dir / "memory.jsonl", rec.to_dict())
- `core/affect/phenomenal_integration.py:596` [affect] `memory.set_write_weights` - memory.set_write_weights(state.memory_weights)
- `core/agency/autonomous_task_engine.py:1007` [agency] `self._mycelial.add_edge` - await self._mycelial.add_edge(context["source_memory"], goal[:40])
- `core/agency/latent_distiller.py:63` [agency] `self.memory.store_memory` - await self.memory.store_memory(
- `core/architect/code_graph.py:679` [architect] `effects.add` - effects.add("memory_write")
- `core/architect/safe_boot_harness.py:79` [architect] `probe_memory_write_read` - memory = await probe_memory_write_read(tmp_root=root / "memory")
- `core/architect/smell_detector.py:177` [architect] `self._effect_smell` - smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, F
- `core/architect/smell_detector.py:177` [architect] `smells.append` - smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, F
- `core/autonomous_initiative_loop.py:1117` [core_root] `memory.store` - await memory.store(
- `core/autonomous_initiative_loop.py:1127` [core_root] `logger.debug` - logger.debug("Social observation memory write failed: %s", exc)
- `core/autonomy/autonomous_research_orchestrator.py:155` [autonomy] `MemoryPersister` - self._persister = persister or MemoryPersister()
- `core/autonomy/initiative_overflow.py:156` [autonomy] `logger.debug` - logger.debug("Skill gap memory write failed: %s", exc)
- `core/autonomy/initiative_overflow.py:166` [autonomy] `memory.store_sync` - memory.store_sync(
- `core/autonomy/personhood_engine.py:199` [autonomy] `state.cognition.working_memory.append` - state.cognition.working_memory.append(
- `core/autonomy/research_cycle.py:695` [autonomy] `state.cognition.long_term_memory.append` - state.cognition.long_term_memory.append(
- `core/autonomy/research_cycle.py:713` [autonomy] `hasattr` - if memory_facade is not None and hasattr(memory_facade, "add_memory"):

### State mutation

Calls that can mutate runtime, identity, repository, or persistent state.

Review candidates:
- `core/adaptation/adaptive_immunity.py:952` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:1299` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:1483` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:1581` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:2305` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:2623` [adaptation] `atomic_write_text` - atomic_write_text(self._state_path, json.dumps(payload, indent=2), encoding="utf-8")
- `core/adaptation/autonomous_resilience.py:326` [adaptation] `set` - registered_names = set(registry.keys())
- `core/adaptation/dream_journal.py:290` [adaptation] `identity_ledger.commitments.all` - for c in identity_ledger.commitments.all()[-10:]
- `core/adaptation/meta_learner.py:300` [adaptation] `np.savez_compressed` - np.savez_compressed(str(_STATE_PATH), **save_dict)
- `core/adaptation/value_autopoiesis.py:142` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:234` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:291` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:512` [adaptation] `os.replace` - os.replace(tmp_path, _STATE_PATH)
- `core/advanced_cognition/integration.py:132` [advanced_cognition] `next_state.setdefault` - next_state.setdefault("_advanced_prediction", {})[act.action_id] = pred
- `core/advanced_cognition/integration.py:223` [advanced_cognition] `issubset` - if isinstance(value, Mapping) and {"domain", "state"}.issubset(value.keys()):
- `core/advanced_cognition/ontology_invention.py:156` [advanced_cognition] `self.save` - self.save(self.state_path)
- `core/advanced_cognition/world_model.py:73` [advanced_cognition] `self.save` - self.save(self.state_path)
- `core/advanced_cognition/zero_shot_transfer.py:75` [advanced_cognition] `self.save` - self.save(self.state_path)
- `core/affect/phenomenal_integration.py:596` [affect] `memory.set_write_weights` - memory.set_write_weights(state.memory_weights)
- `core/agency/agency_core.py:188` [agency] `get_registry.update` - await get_registry().update(active_shards=len(self.active_shards))
- `core/agency/agency_core.py:195` [agency] `setattr` - on_unscheduled=lambda: setattr(self, "_registry_shards_update_pending", False),
- `core/agency/agency_core.py:924` [agency] `virtual_body.__dict__.update` - virtual_body.__dict__.update(snapshot)
- `core/agency/agency_core.py:1118` [agency] `get_registry.update` - await get_registry().update(
- `core/agency/agency_core.py:1127` [agency] `_run_registry_update` - _run_registry_update(),
- `core/agency/agency_core.py:1129` [agency] `setattr` - on_unscheduled=lambda: setattr(self, "_registry_update_pending", False),

### Tool execution

Calls that can execute tools, skills, shells, browsers, or external actions.

Review candidates:
- `core/actuators/actuator_registry.py:375` [actuators] `self.operator.execute_synthesized_tool` - res = self.operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/code_execution_actuator.py:98` [actuators] `operator.execute_synthesized_tool` - res = operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/web_actuators.py:114` [actuators] `skill.execute` - return await skill.execute({"mode": "browse", "url": url}, {})
- `core/agency/agency_core.py:517` [agency] `self._execute_shard_tool` - tasks.append(self._execute_shard_tool(name, payload))
- `core/agency/agency_orchestrator.py:369` [agency] `execute` - await execute(proposal, state_snapshot, receipt.capability_token or "")
- `core/agency/autonomous_task_engine.py:515` [agency] `orchestrator.execute_tool` - return await orchestrator.execute_tool(tool_name, args, **kwargs)
- `core/agency/autonomous_task_engine.py:2841` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:2844` [agency] `orch.execute_tool` - return await orch.execute_tool("web_search", {"query": query})
- `core/agency/autonomous_task_engine.py:2859` [agency] `orch.execute_tool` - result = await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:2863` [agency] `orch.execute_tool` - result = await orch.execute_tool("run_python", {"code": code})
- `core/agency/desktop_planner.py:57` [agency] `skill.execute` - await skill.execute({"action": action, **params}, {})
- `core/agency/skill_library.py:199` [agency] `tool_orchestrator.execute_tool` - result = await tool_orchestrator.execute_tool(step.tool_name, resolved_args)
- `core/agi/curiosity_explorer.py:240` [agi] `orchestrator.execute_tool` - orchestrator.execute_tool(
- `core/autonomous_initiative_loop.py:635` [core_root] `capability_engine.execute` - scan_result = await capability_engine.execute(
- `core/autonomous_initiative_loop.py:681` [core_root] `capability_engine.execute` - test_result = await capability_engine.execute(
- `core/autonomous_initiative_loop.py:720` [core_root] `capability_engine.execute` - proposal_result = await capability_engine.execute(
- `core/autonomous_initiative_loop.py:1094` [core_root] `skill.execute` - return await skill.execute(EmailInput(**payload), {})
- `core/autonomous_initiative_loop.py:1106` [core_root] `skill.execute` - return await skill.execute(RedditInput(**payload), {})
- `core/autonomy/research_cycle.py:540` [autonomy] `self.orchestrator.execute_tool` - lambda name=tool_name, **kw: self.orchestrator.execute_tool(name, kw, origin="research_cycle")
- `core/autonomy/research_cycle.py:547` [autonomy] `self.orchestrator.execute_tool` - lambda name=tool_name, **kw: self.orchestrator.execute_tool(name, kw, origin="research_cycle")
- `core/autonomy/research_cycle.py:844` [autonomy] `self.orchestrator.execute_tool` - result = await self.orchestrator.execute_tool(
- `core/behavior_controller.py:99` [core_root] `self.orchestrator.execute_tool` - self.orchestrator.execute_tool(tool_name, arguments), target_loop
- `core/behavior_controller.py:104` [core_root] `self.orchestrator.execute_tool` - self.orchestrator.execute_tool(tool_name, arguments)
- `core/brain/llm/function_calling_adapter.py:131` [brain] `skill.execute` - result = await skill.execute(args, {"source": "autonomous_brain"})
- `core/brain/llm/local_agent_client.py:314` [brain] `self.adapter.execute_tool` - result_str = await self.adapter.execute_tool(tool_name, tool_args)

### Self-modification and patching

Calls that can generate, validate, apply, or promote code changes.

Review candidates:
- `core/architect/governor.py:140` [architect] `self.promotion_governor.promote` - decision = self.promotion_governor.promote(plan, shadow, proof, rollback)
- `core/factory/software_factory.py:115` [factory] `self.writer.write_patch` - patch = await self.writer.write_patch(change, repo_path)
- `core/guardians/airlock.py:81` [guardians] `atomic_write_text` - atomic_write_text(patch_file, diff_patch, encoding="utf-8")
- `core/kernel/upgrades_10x.py:344` [kernel] `self._safe_self_modify` - await self._safe_self_modify(state)
- `core/optimizer.py:56` [core_root] `patch.apply` - success = await patch.apply(signature)
- `core/optimizer.py:67` [core_root] `cog_patch.apply` - if await cog_patch.apply(signature):
- `core/orchestrator/mixins/boot/boot_autonomy.py:919` [orchestrator] `apply_presence_patch` - apply_presence_patch(self)
- `core/safe_mode.py:140` [core_root] `apply_orchestrator_patches` - apply_orchestrator_patches(orchestrator, safe_mode=bool(enabled))
- `core/security/immune_system.py:259` [security] `self._apply_patch` - reversible_ref = self._apply_patch(ev)
- `core/skill_management/hephaestus.py:198` [skill_management] `guard.validate` - if not guard.validate(patched_code):
- `core/state/cellular_substrate.py:64` [state] `self._apply_patch_recursive` - self._apply_patch_recursive(state, patch)
- `core/state/cellular_substrate.py:82` [state] `self._apply_patch_recursive` - self._apply_patch_recursive(sub_target, value)
- `core/swarm/worker_pool.py:114` [swarm] `writer.write_patch` - patch_res = await writer.write_patch(task_payload.get("change", {}), task_payload.get("repo_path", "."))

### LLM inference

Calls that can spend model context or produce model-authored text/code.

Review candidates:
- `core/actuators/actuator_synthesis.py:157` [actuators] `brain.generate` - res = await brain.generate(prompt, system_prompt=system_prompt)
- `core/adaptation/distillation_pipe.py:104` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/distillation_pipe.py:146` [adaptation] `router.think` - response = await router.think(
- `core/adaptation/dream_journal.py:165` [adaptation] `self.brain.think` - res = await self.brain.think(
- `core/adaptation/epistemic_humility.py:146` [adaptation] `llm.chat` - response = await llm.chat(
- `core/adaptation/heuristic_synthesizer.py:130` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/star_reasoner.py:365` [adaptation] `llm.think` - result = await asyncio.wait_for(llm.think(prompt), timeout=self.RATIONALIZATION_TIMEOUT)
- `core/affect/affective_resonance.py:105` [affect] `brain.think` - brain.think(
- `core/agency/agency_core.py:397` [agency] `structured_brain.generate` - shard_res = await structured_brain.generate(prompt, context=context)
- `core/agency/autonomous_task_engine.py:962` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2593` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2627` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2711` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2820` [agency] `llm.think` - raw = await llm.think(
- `core/agency/latent_distiller.py:49` [agency] `brain.think` - summary = await brain.think(
- `core/agi/curiosity_explorer.py:310` [agi] `router.think` - router.think(prompt, priority=0.3, is_background=True,
- `core/agi/hierarchical_planner.py:215` [agi] `router.think` - router.think(prompt, priority=0.3, is_background=True,
- `core/agi/skill_synthesizer.py:174` [agi] `router.think` - router.think(prompt, priority=0.2, is_background=True,
- `core/audits/alignment_auditor.py:44` [audits] `self.brain.think` - response = await self.brain.think(
- `core/audits/alignment_auditor.py:99` [audits] `self.brain.think` - response = await self.brain.think(
- `core/audits/tool_auditor.py:85` [audits] `self.brain.think` - thought = await self.brain.think(
- `core/autonomy/genuine_refusal.py:449` [autonomy] `llm.think` - llm.think(prompt, mode="FAST"),
- `core/autonomy/genuine_refusal.py:491` [autonomy] `llm.think` - llm.think(prompt, mode="FAST"),
- `core/autonomy/genuine_refusal.py:518` [autonomy] `llm.think` - llm.think(prompt, mode="FAST"),
- `core/autonomy/personhood_engine.py:187` [autonomy] `llm.think` - llm.think(f"[Spontaneous Thought Prompt] {prompt}", mode="FAST"),

### External I/O

Calls that can touch network, subprocesses, sockets, browsers, or APIs.

Review candidates:
- `core/actuators/web_actuators.py:88` [actuators] `urllib.parse.urlparse` - parsed = urllib.parse.urlparse(url)
- `core/agency/tool_orchestrator.py:214` [agency] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/api_adapter.py:105` [core_root] `aiohttp.ClientSession` - self._http_session = aiohttp.ClientSession(
- `core/api_adapter.py:106` [core_root] `aiohttp.TCPConnector` - connector=aiohttp.TCPConnector(limit=100, keepalive_timeout=60)
- `core/autonomic/iot_bridge.py:40` [autonomic] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/autonomic/iot_bridge.py:89` [autonomic] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/bus/sensory_gate.py:209` [bus] `urllib.parse.quote` - f"&search={urllib.parse.quote(query)}&limit=3&namespace=0&format=json"
- `core/capabilities/web_interlocutor.py:150` [capabilities] `str` - self._target_ws_url = str(target.get("webSocketDebuggerUrl") or "")
- `core/capabilities/web_interlocutor.py:150` [capabilities] `target.get` - self._target_ws_url = str(target.get("webSocketDebuggerUrl") or "")
- `core/capabilities/web_interlocutor.py:152` [capabilities] `RuntimeError` - raise RuntimeError("Chrome CDP target did not expose a websocket debugger URL")
- `core/capabilities/web_interlocutor.py:258` [capabilities] `str` - self._target_ws_url = str(target.get("webSocketDebuggerUrl") or "")
- `core/capabilities/web_interlocutor.py:258` [capabilities] `target.get` - self._target_ws_url = str(target.get("webSocketDebuggerUrl") or "")
- `core/capabilities/web_interlocutor.py:264` [capabilities] `urllib.parse.quote` - quoted = urllib.parse.quote(target_url, safe=":/?&=%#")
- `core/capabilities/web_interlocutor.py:335` [capabilities] `RuntimeError` - raise RuntimeError("websocket-client is required for Chrome CDP control") from exc
- `core/capabilities/web_interlocutor.py:336` [capabilities] `websocket.create_connection` - ws = websocket.create_connection(self._target_ws_url, timeout=self.timeout)
- `core/collective/belief_sync.py:201` [collective] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/collective/belief_sync.py:231` [collective] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/collective/belief_sync.py:288` [collective] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/collective/belief_sync.py:382` [collective] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/collective/swarm_protocol.py:26` [collective] `socket.gethostname` - self.node_id = socket.gethostname()
- `core/collective/swarm_protocol.py:52` [collective] `logger.warning` - logger.warning("🕸️ Mycelial Swarm running in offline-only mode; socket binding unavailable.")
- `core/consciousness/heartbeat.py:185` [consciousness] `socket.socket` - with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
- `core/consciousness/heartbeat.py:193` [consciousness] `logger.debug` - logger.debug("Failed to emit keep-alive socket message to watchdog: %s", e)
- `core/device_discovery.py:205` [core_root] `socket.socket` - with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
- `core/device_discovery.py:239` [core_root] `socket.gethostbyaddr` - hostname = socket.gethostbyaddr(device["ip"])[0]

## Degradation Handling

- Total `record_degradation()` calls: 3328
- Log-and-limp candidates: 3000
- Nearby fail-closed candidates: 328

Top limp-on files:

- `core/brain/llm/context_assembler.py`: 31
- `core/brain/inference_gate.py`: 30
- `core/consciousness/consciousness_bridge.py`: 29
- `core/brain/cognitive_engine.py`: 27
- `core/memory/memory_facade.py`: 26
- `core/resilience/memory_governor.py`: 25
- `core/senses/voice_engine.py`: 25
- `core/memory/episodic_memory.py`: 24
- `core/runtime/runtime_hygiene.py`: 23
- `core/capabilities/__init__.py`: 22

## Non-Runtime Candidates

- `core/architect/proof_obligations.py`
- `core/autonomy/autonomous_research_orchestrator.py`
- `core/autonomy/research_cycle.py`
- `core/autonomy/research_triggers.py`
- `core/brain/narrative_memory.py`
- `core/consciousness/animal_cognition.py`
- `core/consciousness/narrative_gravity.py`
- `core/consciousness/oscillatory_binding.py`
- `core/environment/experimentation.py`
- `core/evaluation/behavioral_proof.py`
- `core/factory/repo_cartographer.py`
- `core/lab/experiment_designer.py`
- `core/lab/research_lab.py`
- `core/lab/research_memory.py`
- `core/learning/proof_obligations.py`
- `core/memory/hippocampus.py`
- `core/narrative_thread.py`
- `core/reasoning/proof_answer_solver.py`
- `core/reproducibility/proof_substrate.py`
- `core/runtime/proof_kernel_bridge.py`
- `core/runtime/proof_policy.py`
- `core/search/research_pipeline.py`
- `core/skills/deep_research.py`

## Consolidation Candidates

- `core/audits/`: 2 file(s), 267 line(s)
- `core/coherence/`: 2 file(s), 397 line(s)
- `core/consent/`: 2 file(s), 167 line(s)
- `core/constitution/`: 1 file(s), 25 line(s)
- `core/control/`: 2 file(s), 586 line(s)
- `core/creativity/`: 2 file(s), 801 line(s)
- `core/ethics/`: 2 file(s), 580 line(s)
- `core/evals/`: 2 file(s), 280 line(s)
- `core/initializers/`: 2 file(s), 140 line(s)
- `core/intent/`: 1 file(s), 68 line(s)
- `core/latent/`: 1 file(s), 56 line(s)
- `core/llm/`: 2 file(s), 19 line(s)
- `core/maintenance/`: 2 file(s), 265 line(s)
- `core/media/`: 2 file(s), 273 line(s)
- `core/middleware/`: 2 file(s), 254 line(s)
- `core/multimodal/`: 2 file(s), 185 line(s)
- `core/networking/`: 1 file(s), 318 line(s)
- `core/ontology/`: 2 file(s), 169 line(s)
- `core/persistence/`: 2 file(s), 617 line(s)
- `core/play/`: 1 file(s), 228 line(s)
- `core/predictive/`: 2 file(s), 186 line(s)
- `core/reproducibility/`: 2 file(s), 497 line(s)
- `core/resource/`: 2 file(s), 430 line(s)
- `core/science/`: 1 file(s), 139 line(s)
- `core/search/`: 2 file(s), 1725 line(s)
- `core/sensors/`: 1 file(s), 159 line(s)
- `core/services/`: 2 file(s), 31 line(s)
- `core/session/`: 2 file(s), 231 line(s)
- `core/skill_management/`: 1 file(s), 367 line(s)
- `core/startup/`: 2 file(s), 330 line(s)
- `core/telemetry/`: 2 file(s), 191 line(s)
- `core/transparency/`: 2 file(s), 317 line(s)
- `core/twins/`: 1 file(s), 97 line(s)
