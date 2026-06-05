# Aura Architecture Dependency Map

Schema: `aura.architecture.dependency_map.v2`
Root: `<AURA_ROOT>`
Generated: `1780642824.376558`

## Summary

- Subsystems: 141
- Python files: 1734
- Python lines: 488679
- Dependency edges: 734
- ServiceContainer `.get()` calls: 1490
- ServiceContainer registrations: 358
- Boot contract: PASS

## Subsystem Dependency Graph

```mermaid
graph TD
    runtime["runtime<br/>102 files, 23205 lines"]
    utils["utils<br/>42 files, 5210 lines"]
    brain["brain<br/>116 files, 43920 lines"]
    memory["memory<br/>77 files, 19061 lines"]
    consciousness["consciousness<br/>128 files, 59758 lines"]
    resilience["resilience<br/>53 files, 11331 lines"]
    health["health<br/>3 files, 850 lines"]
    agency["agency<br/>28 files, 14386 lines"]
    adaptation["adaptation<br/>26 files, 12051 lines"]
    affect["affect<br/>6 files, 2935 lines"]
    constitution["constitution<br/>1 files, 25 lines"]
    self_modification["self_modification<br/>29 files, 10905 lines"]
    senses["senses<br/>23 files, 5144 lines"]
    state["state<br/>6 files, 3468 lines"]
    governance["governance<br/>8 files, 2827 lines"]
    observability["observability<br/>3 files, 575 lines"]
    security["security<br/>17 files, 4638 lines"]
    identity["identity<br/>11 files, 2138 lines"]
    orchestrator["orchestrator<br/>42 files, 18872 lines"]
    world_model["world_model<br/>9 files, 2556 lines"]
    executive["executive<br/>4 files, 2342 lines"]
    phases["phases<br/>29 files, 17310 lines"]
    tasks["tasks<br/>3 files, 335 lines"]
    learning["learning<br/>20 files, 6977 lines"]
    actuators["actuators<br/>9 files, 2123 lines"]
    autonomy["autonomy<br/>22 files, 7714 lines"]
    being["being<br/>18 files, 4201 lines"]
    conversation["conversation<br/>8 files, 4300 lines"]
    reasoning["reasoning<br/>7 files, 3893 lines"]
    skills["skills<br/>77 files, 16646 lines"]
    autonomic["autonomic<br/>4 files, 882 lines"]
    coordinators["coordinators<br/>9 files, 4247 lines"]
    managers["managers<br/>6 files, 943 lines"]
    meta["meta<br/>7 files, 1267 lines"]
    ops["ops<br/>11 files, 2355 lines"]
    organism["organism<br/>1 files, 476 lines"]
    self["self<br/>6 files, 1681 lines"]
    supervisor["supervisor<br/>3 files, 527 lines"]
    unity["unity<br/>11 files, 2409 lines"]
    agi["agi<br/>6 files, 1520 lines"]
    cognitive["cognitive<br/>11 files, 8103 lines"]
    collective["collective<br/>6 files, 2006 lines"]
    embodiment["embodiment<br/>15 files, 2646 lines"]
    ethics["ethics<br/>1 files, 310 lines"]
    evaluation["evaluation<br/>10 files, 1768 lines"]
    kernel["kernel<br/>11 files, 6013 lines"]
    motivation["motivation<br/>7 files, 1194 lines"]
    promotion["promotion<br/>6 files, 936 lines"]
    resource["resource<br/>2 files, 430 lines"]
    voice["voice<br/>7 files, 2966 lines"]
    world["world<br/>13 files, 1059 lines"]
    cognition["cognition<br/>9 files, 3465 lines"]
    conversational["conversational<br/>4 files, 2239 lines"]
    data["data<br/>2 files, 514 lines"]
    db["db<br/>4 files, 583 lines"]
    goals["goals<br/>7 files, 3047 lines"]
    morphogenesis["morphogenesis<br/>12 files, 2870 lines"]
    pneuma["pneuma<br/>7 files, 1224 lines"]
    sandbox["sandbox<br/>4 files, 605 lines"]
    search["search<br/>2 files, 1723 lines"]
    somatic["somatic<br/>5 files, 2383 lines"]
    verification["verification<br/>4 files, 350 lines"]
    advanced_cognition["advanced_cognition<br/>13 files, 2905 lines"]
    architect["architect<br/>25 files, 5737 lines"]
    bus["bus<br/>4 files, 2064 lines"]
    coherence["coherence<br/>2 files, 397 lines"]
    discovery["discovery<br/>4 files, 579 lines"]
    environment["environment<br/>82 files, 8309 lines"]
    environments["environments<br/>7 files, 748 lines"]
    evolution["evolution<br/>6 files, 1896 lines"]
    introspection["introspection<br/>3 files, 738 lines"]
    lattice["lattice<br/>5 files, 704 lines"]
    maintenance["maintenance<br/>2 files, 261 lines"]
    perception["perception<br/>15 files, 3264 lines"]
    persistence["persistence<br/>2 files, 617 lines"]
    predictive["predictive<br/>2 files, 186 lines"]
    self_improvement["self_improvement<br/>12 files, 2287 lines"]
    sensors["sensors<br/>1 files, 151 lines"]
    services["services<br/>2 files, 31 lines"]
    simulation["simulation<br/>3 files, 393 lines"]
    social["social<br/>11 files, 3850 lines"]
    soma["soma<br/>3 files, 513 lines"]
    sovereign["sovereign<br/>4 files, 549 lines"]
    startup["startup<br/>2 files, 326 lines"]
    workspace["workspace<br/>3 files, 1069 lines"]
    audit["audit<br/>6 files, 537 lines"]
    consent["consent<br/>2 files, 167 lines"]
    context["context<br/>4 files, 1215 lines"]
    creativity["creativity<br/>2 files, 801 lines"]
    curriculum["curriculum<br/>7 files, 657 lines"]
    cybernetics["cybernetics<br/>6 files, 1133 lines"]
    epistemics["epistemics<br/>7 files, 591 lines"]
    grounding["grounding<br/>7 files, 1092 lines"]
    guardians["guardians<br/>5 files, 625 lines"]
    llm["llm<br/>2 files, 19 lines"]
    media["media<br/>2 files, 273 lines"]
    middleware["middleware<br/>2 files, 254 lines"]
    networking["networking<br/>1 files, 327 lines"]
    phenomenal_substrate["phenomenal_substrate<br/>10 files, 705 lines"]
    plasticity["plasticity<br/>4 files, 342 lines"]
    research_core["research_core<br/>5 files, 580 lines"]
    safety["safety<br/>3 files, 629 lines"]
    skill_management["skill_management<br/>1 files, 367 lines"]
    sovereignty["sovereignty<br/>3 files, 885 lines"]
    transparency["transparency<br/>2 files, 317 lines"]
    twins["twins<br/>1 files, 97 lines"]
    unknowns["unknowns<br/>4 files, 325 lines"]
    actuation["actuation<br/>9 files, 331 lines"]
    adapters["adapters<br/>3 files, 402 lines"]
    audits["audits<br/>2 files, 267 lines"]
    body["body<br/>1 files, 139 lines"]
    control["control<br/>2 files, 586 lines"]
    core_root["core_root<br/>178 files, 54997 lines"]
    council["council<br/>5 files, 466 lines"]
    distributed["distributed<br/>3 files, 140 lines"]
    evals["evals<br/>1 files, 143 lines"]
    factory["factory<br/>8 files, 682 lines"]
    forge["forge<br/>8 files, 325 lines"]
    initializers["initializers<br/>2 files, 140 lines"]
    intent["intent<br/>1 files, 68 lines"]
    knowledge["knowledge<br/>6 files, 142 lines"]
    lab["lab<br/>7 files, 378 lines"]
    latent["latent<br/>1 files, 56 lines"]
    mission["mission<br/>4 files, 472 lines"]
    multimodal["multimodal<br/>2 files, 185 lines"]
    neuroweb["neuroweb<br/>5 files, 368 lines"]
    ontology["ontology<br/>2 files, 169 lines"]
    pipeline["pipeline<br/>3 files, 217 lines"]
    planning["planning<br/>1 files, 137 lines"]
    play["play<br/>1 files, 228 lines"]
    providers["providers<br/>6 files, 1208 lines"]
    reproducibility["reproducibility<br/>2 files, 497 lines"]
    science["science<br/>1 files, 71 lines"]
    session["session<br/>2 files, 231 lines"]
    sim["sim<br/>5 files, 219 lines"]
    swarm["swarm<br/>5 files, 360 lines"]
    systems["systems<br/>3 files, 256 lines"]
    telemetry["telemetry<br/>2 files, 191 lines"]
    temporal["temporal<br/>3 files, 1507 lines"]
    tools["tools<br/>9 files, 863 lines"]
    values["values<br/>2 files, 289 lines"]
    runtime --> actuators
    runtime --> adaptation
    runtime --> agency
    runtime --> architect
    runtime --> autonomy
    runtime --> being
    runtime --> consciousness
    runtime --> constitution
    runtime --> conversation
    runtime --> evaluation
    runtime --> governance
    runtime --> health
    runtime --> identity
    runtime --> memory
    runtime --> observability
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
    brain --> conversation
    brain --> health
    brain --> identity
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
    memory --> constitution
    memory --> db
    memory --> governance
    memory --> health
    memory --> phases
    memory --> resilience
    memory --> runtime
    memory --> utils
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
    agency --> adaptation
    agency --> affect
    agency --> agi
    agency --> brain
    agency --> consciousness
    agency --> constitution
    agency --> governance
    agency --> health
    agency --> identity
    agency --> orchestrator
    agency --> organism
    agency --> resilience
    agency --> runtime
    agency --> tasks
    agency --> utils
    adaptation --> actuators
    adaptation --> affect
    adaptation --> brain
    adaptation --> cognitive
    adaptation --> health
    adaptation --> learning
    adaptation --> memory
    adaptation --> resilience
    adaptation --> runtime
    adaptation --> sensors
    adaptation --> utils
    adaptation --> world
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
    senses --> resilience
    senses --> runtime
    senses --> security
    senses --> supervisor
    senses --> utils
    state --> constitution
    state --> governance
    state --> memory
    state --> motivation
    state --> runtime
    state --> unity
    state --> utils
    governance --> actuators
    governance --> being
    governance --> consciousness
    governance --> memory
    governance --> runtime
    observability --> runtime
    security --> affect
    security --> agency
    security --> consciousness
    security --> identity
    security --> memory
    security --> runtime
    security --> utils
    identity --> agency
    identity --> brain
    identity --> governance
    identity --> organism
    identity --> runtime
    identity --> utils
    orchestrator --> adaptation
    orchestrator --> affect
    orchestrator --> agency
    orchestrator --> agi
    orchestrator --> audit
    orchestrator --> autonomic
    orchestrator --> autonomy
    orchestrator --> brain
    orchestrator --> bus
    orchestrator --> cognitive
    orchestrator --> collective
    orchestrator --> consciousness
    orchestrator --> constitution
    orchestrator --> conversation
    orchestrator --> coordinators
    orchestrator --> data
    orchestrator --> db
    orchestrator --> embodiment
    orchestrator --> environment
    orchestrator --> evolution
    orchestrator --> executive
    orchestrator --> guardians
    orchestrator --> health
    orchestrator --> identity
    orchestrator --> kernel
    orchestrator --> learning
    orchestrator --> maintenance
    orchestrator --> managers
    orchestrator --> memory
    orchestrator --> meta
    orchestrator --> morphogenesis
    orchestrator --> motivation
    orchestrator --> observability
    orchestrator --> ops
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
    world_model --> brain
    world_model --> constitution
    world_model --> health
    world_model --> resilience
    world_model --> runtime
    executive --> agency
    executive --> autonomy
    executive --> consciousness
    executive --> constitution
    executive --> goals
    executive --> health
    executive --> memory
    executive --> runtime
    executive --> state
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
    phases --> reasoning
    phases --> runtime
    phases --> self_modification
    phases --> skills
    phases --> somatic
    phases --> state
    phases --> unity
    phases --> utils
    phases --> voice
    tasks --> runtime
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
    actuators --> affect
    actuators --> brain
    actuators --> executive
    actuators --> memory
    actuators --> runtime
    actuators --> sandbox
    actuators --> search
    actuators --> skills
    actuators --> world
    autonomy --> affect
    autonomy --> agency
    autonomy --> consciousness
    autonomy --> executive
    autonomy --> memory
    autonomy --> observability
    autonomy --> resource
    autonomy --> runtime
    autonomy --> state
    autonomy --> utils
    being --> runtime
    conversation --> brain
    conversation --> consciousness
    conversation --> memory
    conversation --> runtime
    conversation --> social
    conversation --> utils
    reasoning --> runtime
    skills --> actuators
    skills --> advanced_cognition
    skills --> affect
    skills --> brain
    skills --> consent
    skills --> embodiment
    skills --> executive
    skills --> governance
    skills --> learning
    skills --> memory
    skills --> runtime
    skills --> sandbox
    skills --> search
    skills --> security
    skills --> self_modification
    skills --> senses
    skills --> sovereign
    skills --> utils
    autonomic --> orchestrator
    autonomic --> runtime
    autonomic --> utils
    coordinators --> autonomy
    coordinators --> brain
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
    organism --> runtime
    organism --> utils
    self --> affect
    self --> bus
    self --> consciousness
    self --> memory
    self --> runtime
    self --> security
    self --> senses
    self --> state
    self --> utils
    supervisor --> runtime
    unity --> consciousness
    unity --> runtime
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
    ethics --> runtime
    evaluation --> learning
    evaluation --> promotion
    evaluation --> runtime
    kernel --> agency
    kernel --> brain
    kernel --> cognition
    kernel --> consciousness
    kernel --> cybernetics
    kernel --> executive
    kernel --> health
    kernel --> learning
    kernel --> orchestrator
    kernel --> phases
    kernel --> resilience
    kernel --> runtime
    kernel --> security
    kernel --> self_modification
    kernel --> senses
    kernel --> somatic
    kernel --> state
    kernel --> utils
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
    voice --> brain
    voice --> conversational
    voice --> phases
    voice --> resilience
    voice --> runtime
    voice --> senses
    voice --> utils
    world --> governance
    world --> runtime
    cognition --> memory
    cognition --> runtime
    cognition --> world_model
    conversational --> memory
    conversational --> runtime
    data --> runtime
    db --> runtime
    goals --> agency
    goals --> runtime
    goals --> state
    morphogenesis --> adaptation
    morphogenesis --> memory
    morphogenesis --> resilience
    morphogenesis --> runtime
    morphogenesis --> self_modification
    morphogenesis --> utils
    pneuma --> affect
    pneuma --> runtime
    pneuma --> utils
    sandbox --> runtime
    search --> runtime
    somatic --> memory
    somatic --> runtime
    somatic --> utils
    verification --> discovery
    verification --> middleware
    advanced_cognition --> reasoning
    advanced_cognition --> runtime
    architect --> adaptation
    architect --> brain
    architect --> consciousness
    architect --> runtime
    architect --> self_modification
    architect --> world_model
    bus --> resilience
    bus --> runtime
    bus --> utils
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
    maintenance --> resilience
    maintenance --> runtime
    perception --> brain
    perception --> runtime
    persistence --> observability
    persistence --> resilience
    persistence --> runtime
    predictive --> brain
    predictive --> runtime
    predictive --> utils
    self_improvement --> brain
    self_improvement --> runtime
    self_improvement --> self_modification
    sensors --> world
    services --> autonomic
    simulation --> brain
    simulation --> consciousness
    simulation --> runtime
    simulation --> world_model
    social --> agency
    social --> ethics
    social --> governance
    social --> runtime
    social --> utils
    soma --> resilience
    soma --> runtime
    soma --> utils
    sovereign --> runtime
    startup --> brain
    startup --> runtime
    startup --> senses
    workspace --> runtime
    audit --> epistemics
    audit --> runtime
    context --> runtime
    creativity --> memory
    creativity --> runtime
    curriculum --> runtime
    cybernetics --> cognitive
    cybernetics --> kernel
    cybernetics --> runtime
    cybernetics --> utils
    grounding --> plasticity
    grounding --> runtime
    guardians --> brain
    guardians --> runtime
    guardians --> tasks
    guardians --> utils
    llm --> brain
    middleware --> runtime
    networking --> runtime
    networking --> utils
    plasticity --> runtime
    research_core --> curriculum
    research_core --> discovery
    research_core --> lattice
    research_core --> promotion
    research_core --> runtime
    research_core --> unknowns
    research_core --> verification
    safety --> runtime
    skill_management --> resilience
    skill_management --> runtime
    skill_management --> self_modification
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
    core_root --> predictive
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
    core_root --> tasks
    core_root --> transparency
    core_root --> utils
    core_root --> voice
    core_root --> workspace
    core_root --> world_model
    council --> runtime
    factory --> runtime
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
    planning --> runtime
    play --> consciousness
    play --> runtime
    providers --> affect
    providers --> brain
    providers --> cognition
    providers --> collective
    providers --> consciousness
    providers --> coordinators
    providers --> creativity
    providers --> db
    providers --> learning
    providers --> managers
    providers --> memory
    providers --> motivation
    providers --> ops
    providers --> orchestrator
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
    session --> runtime
    sim --> twins
    swarm --> runtime
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
| consciousness | 128 | 59758 | 2514202 | 38 | 29 |
| core_root | 178 | 54997 | 2236455 | 99 | 0 |
| brain | 116 | 43920 | 1894429 | 42 | 39 |
| runtime | 102 | 23205 | 819406 | 41 | 117 |
| memory | 77 | 19061 | 769495 | 18 | 32 |
| orchestrator | 42 | 18872 | 832854 | 124 | 9 |
| phases | 29 | 17310 | 783345 | 33 | 8 |
| skills | 77 | 16646 | 672690 | 29 | 6 |
| agency | 28 | 14386 | 583488 | 28 | 17 |
| adaptation | 26 | 12051 | 482023 | 19 | 14 |
| resilience | 53 | 11331 | 456372 | 16 | 25 |
| self_modification | 29 | 10905 | 434070 | 12 | 14 |
| environment | 82 | 8309 | 322052 | 10 | 2 |
| cognitive | 11 | 8103 | 330009 | 9 | 4 |
| autonomy | 22 | 7714 | 316858 | 17 | 6 |
| learning | 20 | 6977 | 276754 | 15 | 7 |
| kernel | 11 | 6013 | 251231 | 22 | 4 |
| architect | 25 | 5737 | 239735 | 9 | 2 |
| utils | 42 | 5210 | 203061 | 17 | 50 |
| senses | 23 | 5144 | 215535 | 16 | 14 |
| security | 17 | 4638 | 182760 | 11 | 10 |
| conversation | 8 | 4300 | 158729 | 11 | 6 |
| coordinators | 9 | 4247 | 199509 | 36 | 5 |
| being | 18 | 4201 | 160068 | 3 | 6 |
| reasoning | 7 | 3893 | 157700 | 2 | 6 |
| social | 11 | 3850 | 163845 | 8 | 2 |
| state | 6 | 3468 | 146953 | 9 | 13 |
| cognition | 9 | 3465 | 140466 | 8 | 3 |
| perception | 15 | 3264 | 130056 | 4 | 2 |
| goals | 7 | 3047 | 129238 | 6 | 3 |
| voice | 7 | 2966 | 134320 | 10 | 4 |
| affect | 6 | 2935 | 136046 | 13 | 14 |
| advanced_cognition | 13 | 2905 | 118305 | 3 | 2 |
| morphogenesis | 12 | 2870 | 111460 | 8 | 3 |
| governance | 8 | 2827 | 116215 | 10 | 12 |
| embodiment | 15 | 2646 | 103210 | 12 | 4 |
| world_model | 9 | 2556 | 103782 | 8 | 9 |
| unity | 11 | 2409 | 100598 | 3 | 5 |
| somatic | 5 | 2383 | 90349 | 8 | 3 |
| ops | 11 | 2355 | 91027 | 15 | 5 |
| executive | 4 | 2342 | 98871 | 14 | 8 |
| self_improvement | 12 | 2287 | 87032 | 3 | 2 |
| conversational | 4 | 2239 | 95619 | 4 | 3 |
| identity | 11 | 2138 | 90683 | 9 | 9 |
| actuators | 9 | 2123 | 83231 | 12 | 6 |
| bus | 4 | 2064 | 84924 | 6 | 2 |
| collective | 6 | 2006 | 82544 | 8 | 4 |
| evolution | 6 | 1896 | 77054 | 8 | 2 |
| evaluation | 10 | 1768 | 62043 | 3 | 4 |
| search | 2 | 1723 | 65416 | 6 | 3 |
| self | 6 | 1681 | 70508 | 12 | 5 |
| agi | 6 | 1520 | 63529 | 13 | 4 |
| temporal | 3 | 1507 | 50941 | 2 | 0 |
| meta | 7 | 1267 | 47564 | 5 | 5 |
| pneuma | 7 | 1224 | 46166 | 4 | 3 |
| context | 4 | 1215 | 47010 | 1 | 1 |
| providers | 6 | 1208 | 53206 | 52 | 0 |
| motivation | 7 | 1194 | 50455 | 10 | 4 |
| cybernetics | 6 | 1133 | 45264 | 6 | 1 |
| grounding | 7 | 1092 | 40453 | 4 | 1 |
| workspace | 3 | 1069 | 39594 | 3 | 2 |
| world | 13 | 1059 | 37631 | 3 | 4 |
| managers | 6 | 943 | 40300 | 25 | 5 |
| promotion | 6 | 936 | 31616 | 1 | 4 |
| sovereignty | 3 | 885 | 33782 | 10 | 1 |
| autonomic | 4 | 882 | 36599 | 5 | 5 |
| tools | 9 | 863 | 31892 | 4 | 0 |
| health | 3 | 850 | 32454 | 7 | 21 |
| creativity | 2 | 801 | 33361 | 3 | 1 |
| environments | 7 | 748 | 31101 | 3 | 2 |
| introspection | 3 | 738 | 28467 | 1 | 2 |
| phenomenal_substrate | 10 | 705 | 27209 | 0 | 1 |
| lattice | 5 | 704 | 26089 | 0 | 2 |
| factory | 8 | 682 | 25812 | 2 | 0 |
| curriculum | 7 | 657 | 21995 | 1 | 1 |
| safety | 3 | 629 | 25738 | 3 | 1 |
| guardians | 5 | 625 | 27444 | 6 | 1 |
| persistence | 2 | 617 | 24953 | 3 | 2 |
| sandbox | 4 | 605 | 21383 | 1 | 3 |
| epistemics | 7 | 591 | 22459 | 0 | 1 |
| control | 2 | 586 | 21056 | 4 | 0 |
| db | 4 | 583 | 22581 | 2 | 3 |
| research_core | 5 | 580 | 22543 | 8 | 1 |
| discovery | 4 | 579 | 20581 | 2 | 2 |
| observability | 3 | 575 | 20634 | 4 | 11 |
| sovereign | 4 | 549 | 19278 | 1 | 2 |
| audit | 6 | 537 | 20846 | 4 | 1 |
| supervisor | 3 | 527 | 19236 | 1 | 5 |
| data | 2 | 514 | 19319 | 2 | 3 |
| soma | 3 | 513 | 19932 | 3 | 2 |
| reproducibility | 2 | 497 | 18141 | 1 | 0 |
| organism | 1 | 476 | 18672 | 3 | 5 |
| mission | 4 | 472 | 17806 | 1 | 0 |
| council | 5 | 466 | 18107 | 3 | 0 |
| resource | 2 | 430 | 15691 | 3 | 4 |
| adapters | 3 | 402 | 13469 | 1 | 0 |
| coherence | 2 | 397 | 18920 | 6 | 2 |
| simulation | 3 | 393 | 15639 | 6 | 2 |
| lab | 7 | 378 | 13494 | 0 | 0 |
| neuroweb | 5 | 368 | 14195 | 5 | 0 |
| skill_management | 1 | 367 | 17964 | 6 | 1 |
| swarm | 5 | 360 | 12068 | 1 | 0 |
| verification | 4 | 350 | 13177 | 2 | 3 |
| plasticity | 4 | 342 | 12056 | 2 | 1 |
| tasks | 3 | 335 | 11520 | 3 | 8 |
| actuation | 9 | 331 | 11312 | 2 | 0 |
| networking | 1 | 327 | 12336 | 3 | 1 |
| startup | 2 | 326 | 11280 | 5 | 2 |
| forge | 8 | 325 | 11877 | 2 | 0 |
| unknowns | 4 | 325 | 11829 | 3 | 1 |
| transparency | 2 | 317 | 12256 | 1 | 1 |
| ethics | 1 | 310 | 11875 | 1 | 4 |
| values | 2 | 289 | 10861 | 0 | 0 |
| media | 2 | 273 | 9349 | 0 | 1 |
| audits | 2 | 267 | 9492 | 3 | 0 |
| maintenance | 2 | 261 | 9351 | 4 | 2 |
| systems | 3 | 256 | 9861 | 3 | 0 |
| middleware | 2 | 254 | 11019 | 2 | 1 |
| session | 2 | 231 | 9389 | 1 | 0 |
| play | 1 | 228 | 8774 | 4 | 0 |
| sim | 5 | 219 | 7359 | 1 | 0 |
| pipeline | 3 | 217 | 6684 | 1 | 0 |
| telemetry | 2 | 191 | 5594 | 0 | 0 |
| predictive | 2 | 186 | 7105 | 5 | 2 |
| multimodal | 2 | 185 | 6591 | 1 | 0 |
| ontology | 2 | 169 | 5381 | 0 | 0 |
| consent | 2 | 167 | 5514 | 0 | 1 |
| sensors | 1 | 151 | 5644 | 1 | 2 |
| evals | 1 | 143 | 4924 | 0 | 0 |
| knowledge | 6 | 142 | 3870 | 0 | 0 |
| distributed | 3 | 140 | 4655 | 0 | 0 |
| initializers | 2 | 140 | 6565 | 10 | 0 |
| body | 1 | 139 | 5237 | 0 | 0 |
| planning | 1 | 137 | 5440 | 2 | 0 |
| twins | 1 | 97 | 3626 | 0 | 1 |
| science | 1 | 71 | 2675 | 0 | 0 |
| intent | 1 | 68 | 2661 | 1 | 0 |
| latent | 1 | 56 | 2337 | 0 | 0 |
| services | 2 | 31 | 1171 | 1 | 2 |
| constitution | 1 | 25 | 795 | 0 | 14 |
| llm | 2 | 19 | 745 | 1 | 1 |

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

- Unique services retrieved: 366
- Unique services registered: 291
- Services retrieved without detected registration: 189

### Top Fetched Services

| Service | Gets | Registrations |
| --- | ---: | ---: |
| orchestrator | 67 | 3 |
| cognitive_engine | 53 | 3 |
| llm_router | 44 | 2 |
| affect_engine | 40 | 1 |
| inference_gate | 38 | 4 |
| capability_engine | 34 | 2 |
| memory_facade | 31 | 1 |
| liquid_substrate | 28 | 1 |
| mycelial_network | 28 | 2 |
| free_energy_engine | 24 | 0 |
| homeostasis | 22 | 1 |
| global_workspace | 22 | 2 |
| drive_engine | 22 | 0 |
| state_repository | 20 | 1 |
| goal_engine | 20 | 0 |
| qualia_synthesizer | 19 | 3 |
| knowledge_graph | 18 | 0 |
| episodic_memory | 18 | 1 |
| belief_revision_engine | 17 | 1 |
| subsystem_audit | 16 | 2 |

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
- `api_adapter` fetched 6 time(s)
- `archive_engine` fetched 3 time(s)
- `audit_log` fetched 1 time(s)
- `audit_suite` fetched 1 time(s)
- `aura_state` fetched 3 time(s)
- `autonomous_loop` fetched 1 time(s)
- `autonomous_resilience_mesh` fetched 1 time(s)
- `autopoiesis` fetched 1 time(s)
- `belief_challenger` fetched 2 time(s)
- `belief_engine` fetched 1 time(s)
- `belief_system` fetched 1 time(s)
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
- `code_refiner` fetched 1 time(s)
- `code_repair` fetched 1 time(s)
- `cognitive_integration_layer` fetched 1 time(s)
- `cognitive_kernel` fetched 3 time(s)
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
- `conversation_intelligence` fetched 1 time(s)
- `conversational_dynamics` fetched 1 time(s)
- `conversational_profiler` fetched 1 time(s)

## Operational Authority Map

| Surface | Calls | Files | Owner Calls | Review Candidates |
| --- | ---: | ---: | ---: | ---: |
| UnifiedWill decisions | 55 | 29 | 2 | 53 |
| Memory writes | 281 | 113 | 48 | 233 |
| State mutation | 376 | 145 | 5 | 371 |
| Tool execution | 93 | 48 | 6 | 87 |
| Self-modification and patching | 13 | 10 | 2 | 11 |
| LLM inference | 253 | 150 | 67 | 186 |
| External I/O | 91 | 42 | 10 | 81 |

### UnifiedWill decisions

Calls that can ask the single will authority to approve action.

Review candidates:
- `core/actuators/actuator_synthesis.py:183` [actuators] `get_will` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:183` [actuators] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/adaptive_immunity.py:1986` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/adaptive_immunity.py:1986` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:624` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:624` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:172` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:172` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/agency/agency_bus.py:83` [agency] `get_will` - _auto_decision = get_will().decide(
- `core/agency/agency_bus.py:83` [agency] `get_will.decide` - _auto_decision = get_will().decide(
- `core/autonomy/self_modification.py:315` [autonomy] `will.decide` - decision = will.decide(
- `core/cognitive/autopoiesis.py:961` [cognitive] `will.decide` - decision = will.decide(
- `core/consciousness/parallel_branches.py:736` [consciousness] `will.decide` - decision = will.decide(
- `core/environment/governance_bridge.py:47` [environment] `self.will_gateway.decide` - will_decision = await self.will_gateway.decide(intent)
- `core/goals/goal_engine.py:1013` [goals] `will.decide` - decision = will.decide(
- `core/governance/will_gate.py:109` [governance] `will.decide` - decision = will.decide(
- `core/governance/will_gate.py:160` [governance] `will.decide` - decision = will.decide(
- `core/initiative_synthesis.py:744` [core_root] `get_will` - decision = get_will().decide(
- `core/initiative_synthesis.py:744` [core_root] `get_will.decide` - decision = get_will().decide(
- `core/learning/genuine_learning_pipeline.py:649` [learning] `get_will` - decision = get_will().decide(
- `core/learning/genuine_learning_pipeline.py:649` [learning] `get_will.decide` - decision = get_will().decide(
- `core/learning/recursive_self_improvement.py:502` [learning] `get_will` - decision = get_will().decide(
- `core/learning/recursive_self_improvement.py:502` [learning] `get_will.decide` - decision = get_will().decide(
- `core/mind_tick.py:106` [core_root] `get_will` - decision = get_will().decide(
- `core/mind_tick.py:106` [core_root] `get_will.decide` - decision = get_will().decide(

### Memory writes

Calls that can create durable or semantically promoted memory.

Review candidates:
- `core/actuators/doc_ingest.py:81` [actuators] `get_memory_write_gateway` - gateway = get_memory_write_gateway()
- `core/actuators/doc_ingest.py:95` [actuators] `MemoryWriteRequest` - MemoryWriteRequest(
- `core/actuators/doc_ingest.py:106` [actuators] `memory_facade.add_memory` - maybe_result = memory_facade.add_memory(text=text, metadata=metadata)
- `core/adaptation/abstraction_engine.py:121` [adaptation] `MemoryWriteReceipt` - MemoryWriteReceipt(
- `core/adaptation/abstraction_engine.py:140` [adaptation] `memory_facade.store` - await memory_facade.store(
- `core/adaptation/adaptive_immunity.py:1230` [adaptation] `self._cells.append` - self._cells.append(memory)
- `core/advanced_cognition/continual_learning_stability.py:94` [advanced_cognition] `self._persist_memory` - self._persist_memory(rec)
- `core/advanced_cognition/continual_learning_stability.py:98` [advanced_cognition] `self.store_memory` - return self.store_memory(
- `core/advanced_cognition/continual_learning_stability.py:208` [advanced_cognition] `scored.append` - scored.append((score, memory))
- `core/advanced_cognition/continual_learning_stability.py:313` [advanced_cognition] `self._append_jsonl` - self._append_jsonl(self.state_dir / "memory.jsonl", rec.to_dict())
- `core/affect/phenomenal_integration.py:565` [affect] `memory.set_write_weights` - memory.set_write_weights(state.memory_weights)
- `core/agency/autonomous_task_engine.py:1007` [agency] `self._mycelial.add_edge` - await self._mycelial.add_edge(context["source_memory"], goal[:40])
- `core/agency/latent_distiller.py:63` [agency] `self.memory.store_memory` - await self.memory.store_memory(
- `core/architect/code_graph.py:679` [architect] `effects.add` - effects.add("memory_write")
- `core/architect/safe_boot_harness.py:79` [architect] `probe_memory_write_read` - memory = await probe_memory_write_read(tmp_root=root / "memory")
- `core/architect/smell_detector.py:177` [architect] `self._effect_smell` - smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, F
- `core/architect/smell_detector.py:177` [architect] `smells.append` - smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, F
- `core/autonomous_initiative_loop.py:955` [core_root] `memory.store` - await memory.store(
- `core/autonomous_initiative_loop.py:965` [core_root] `logger.debug` - logger.debug("Social observation memory write failed: %s", exc)
- `core/autonomy/autonomous_research_orchestrator.py:143` [autonomy] `MemoryPersister` - self._persister = persister or MemoryPersister()
- `core/autonomy/initiative_overflow.py:156` [autonomy] `logger.debug` - logger.debug("Skill gap memory write failed: %s", exc)
- `core/autonomy/initiative_overflow.py:166` [autonomy] `memory.store_sync` - memory.store_sync(
- `core/autonomy/personhood_engine.py:199` [autonomy] `state.cognition.working_memory.append` - state.cognition.working_memory.append(
- `core/autonomy/research_cycle.py:687` [autonomy] `state.cognition.long_term_memory.append` - state.cognition.long_term_memory.append(
- `core/autonomy/research_cycle.py:705` [autonomy] `hasattr` - if memory_facade is not None and hasattr(memory_facade, "add_memory"):

### State mutation

Calls that can mutate runtime, identity, repository, or persistent state.

Review candidates:
- `core/adaptation/adaptive_immunity.py:919` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:1266` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:1446` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:1525` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:2230` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:2548` [adaptation] `atomic_write_text` - atomic_write_text(self._state_path, json.dumps(payload, indent=2), encoding="utf-8")
- `core/adaptation/autonomous_resilience.py:326` [adaptation] `set` - registered_names = set(registry.keys())
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
- `core/affect/phenomenal_integration.py:565` [affect] `memory.set_write_weights` - memory.set_write_weights(state.memory_weights)
- `core/agency/agency_core.py:139` [agency] `get_registry.update` - await get_registry().update(active_shards=len(self.active_shards))
- `core/agency/agency_core.py:841` [agency] `virtual_body.__dict__.update` - virtual_body.__dict__.update(snapshot)
- `core/agency/agency_core.py:1035` [agency] `get_registry.update` - await get_registry().update(
- `core/agency/agency_core.py:1044` [agency] `_run_registry_update` - _run_registry_update(),
- `core/agency/autonomous_task_engine.py:425` [agency] `self._update_state_goals` - self._update_state_goals(plan)
- `core/agency/autonomous_task_engine.py:661` [agency] `self._update_state_goals` - self._update_state_goals(plan)
- `core/agency/autonomous_task_engine.py:681` [agency] `self._update_state_goals` - self._update_state_goals(plan)

### Tool execution

Calls that can execute tools, skills, shells, browsers, or external actions.

Review candidates:
- `core/actuators/actuator_registry.py:375` [actuators] `self.operator.execute_synthesized_tool` - res = self.operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/code_execution_actuator.py:98` [actuators] `operator.execute_synthesized_tool` - res = operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/web_actuators.py:114` [actuators] `skill.execute` - return await skill.execute({"mode": "browse", "url": url}, {})
- `core/agency/agency_core.py:447` [agency] `self._execute_shard_tool` - tasks.append(self._execute_shard_tool(name, payload))
- `core/agency/agency_orchestrator.py:367` [agency] `execute` - await execute(proposal, state_snapshot, receipt.capability_token or "")
- `core/agency/autonomous_task_engine.py:515` [agency] `orchestrator.execute_tool` - return await orchestrator.execute_tool(tool_name, args, **kwargs)
- `core/agency/autonomous_task_engine.py:2844` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:2847` [agency] `orch.execute_tool` - return await orch.execute_tool("web_search", {"query": query})
- `core/agency/autonomous_task_engine.py:2862` [agency] `orch.execute_tool` - result = await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:2866` [agency] `orch.execute_tool` - result = await orch.execute_tool("run_python", {"code": code})
- `core/agency/skill_library.py:199` [agency] `tool_orchestrator.execute_tool` - result = await tool_orchestrator.execute_tool(step.tool_name, resolved_args)
- `core/agi/curiosity_explorer.py:240` [agi] `orchestrator.execute_tool` - orchestrator.execute_tool(
- `core/autonomous_initiative_loop.py:473` [core_root] `capability_engine.execute` - scan_result = await capability_engine.execute(
- `core/autonomous_initiative_loop.py:519` [core_root] `capability_engine.execute` - test_result = await capability_engine.execute(
- `core/autonomous_initiative_loop.py:558` [core_root] `capability_engine.execute` - proposal_result = await capability_engine.execute(
- `core/autonomous_initiative_loop.py:932` [core_root] `skill.execute` - return await skill.execute(EmailInput(**payload), {})
- `core/autonomous_initiative_loop.py:944` [core_root] `skill.execute` - return await skill.execute(RedditInput(**payload), {})
- `core/autonomy/research_cycle.py:532` [autonomy] `self.orchestrator.execute_tool` - lambda name=tool_name, **kw: self.orchestrator.execute_tool(name, kw, origin="research_cycle")
- `core/autonomy/research_cycle.py:539` [autonomy] `self.orchestrator.execute_tool` - lambda name=tool_name, **kw: self.orchestrator.execute_tool(name, kw, origin="research_cycle")
- `core/autonomy/research_cycle.py:836` [autonomy] `self.orchestrator.execute_tool` - result = await self.orchestrator.execute_tool(
- `core/behavior_controller.py:99` [core_root] `self.orchestrator.execute_tool` - self.orchestrator.execute_tool(tool_name, arguments), target_loop
- `core/behavior_controller.py:104` [core_root] `self.orchestrator.execute_tool` - self.orchestrator.execute_tool(tool_name, arguments)
- `core/brain/llm/function_calling_adapter.py:131` [brain] `skill.execute` - result = await skill.execute(args, {"source": "autonomous_brain"})
- `core/brain/llm/local_agent_client.py:310` [brain] `self.adapter.execute_tool` - result_str = await self.adapter.execute_tool(tool_name, tool_args)
- `core/brain/multimodal_orchestrator.py:216` [brain] `execute` - _maybe_await(execute(skill_name, payload)),

### Self-modification and patching

Calls that can generate, validate, apply, or promote code changes.

Review candidates:
- `core/architect/governor.py:140` [architect] `self.promotion_governor.promote` - decision = self.promotion_governor.promote(plan, shadow, proof, rollback)
- `core/factory/software_factory.py:115` [factory] `self.writer.write_patch` - patch = await self.writer.write_patch(change, repo_path)
- `core/guardians/airlock.py:81` [guardians] `atomic_write_text` - atomic_write_text(patch_file, diff_patch, encoding="utf-8")
- `core/kernel/upgrades_10x.py:335` [kernel] `self._safe_self_modify` - await self._safe_self_modify(state)
- `core/optimizer.py:61` [core_root] `patch.apply` - success = await patch.apply(signature)
- `core/optimizer.py:63` [core_root] `patch.apply` - success = await patch.apply()
- `core/optimizer.py:74` [core_root] `cog_patch.apply` - if await cog_patch.apply(signature):
- `core/orchestrator/mixins/boot/boot_autonomy.py:870` [orchestrator] `apply_presence_patch` - apply_presence_patch(self)
- `core/skill_management/hephaestus.py:198` [skill_management] `guard.validate` - if not guard.validate(patched_code):
- `core/state/cellular_substrate.py:64` [state] `self._apply_patch_recursive` - self._apply_patch_recursive(state, patch)
- `core/state/cellular_substrate.py:82` [state] `self._apply_patch_recursive` - self._apply_patch_recursive(sub_target, value)

### LLM inference

Calls that can spend model context or produce model-authored text/code.

Review candidates:
- `core/actuators/actuator_synthesis.py:157` [actuators] `brain.generate` - res = await brain.generate(prompt, system_prompt=system_prompt)
- `core/adaptation/distillation_pipe.py:104` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/distillation_pipe.py:146` [adaptation] `router.think` - response = await router.think(
- `core/adaptation/dream_journal.py:162` [adaptation] `self.brain.think` - res = await self.brain.think(
- `core/adaptation/epistemic_humility.py:146` [adaptation] `llm.chat` - response = await llm.chat(
- `core/adaptation/heuristic_synthesizer.py:130` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/star_reasoner.py:365` [adaptation] `llm.think` - result = await asyncio.wait_for(llm.think(prompt), timeout=self.RATIONALIZATION_TIMEOUT)
- `core/agency/agency_core.py:327` [agency] `structured_brain.generate` - shard_res = await structured_brain.generate(prompt, context=context)
- `core/agency/autonomous_task_engine.py:962` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2596` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2630` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2714` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2823` [agency] `llm.think` - raw = await llm.think(
- `core/agency/latent_distiller.py:49` [agency] `brain.think` - summary = await brain.think(
- `core/agi/curiosity_explorer.py:310` [agi] `router.think` - router.think(prompt, priority=0.3, is_background=True,
- `core/agi/hierarchical_planner.py:215` [agi] `router.think` - router.think(prompt, priority=0.3, is_background=True,
- `core/agi/skill_synthesizer.py:174` [agi] `router.think` - router.think(prompt, priority=0.2, is_background=True,
- `core/audits/alignment_auditor.py:44` [audits] `self.brain.think` - response = await self.brain.think(
- `core/audits/alignment_auditor.py:99` [audits] `self.brain.think` - response = await self.brain.think(
- `core/audits/tool_auditor.py:85` [audits] `self.brain.think` - thought = await self.brain.think(
- `core/autonomy/genuine_refusal.py:334` [autonomy] `llm.think` - llm.think(prompt, mode="FAST"),
- `core/autonomy/genuine_refusal.py:376` [autonomy] `llm.think` - llm.think(prompt, mode="FAST"),
- `core/autonomy/genuine_refusal.py:403` [autonomy] `llm.think` - llm.think(prompt, mode="FAST"),
- `core/autonomy/personhood_engine.py:187` [autonomy] `llm.think` - llm.think(f"[Spontaneous Thought Prompt] {prompt}", mode="FAST"),
- `core/autonomy/research_cycle.py:579` [autonomy] `llm.think` - return await llm.think(prompt)

### External I/O

Calls that can touch network, subprocesses, sockets, browsers, or APIs.

Review candidates:
- `core/actuators/web_actuators.py:88` [actuators] `urllib.parse.urlparse` - parsed = urllib.parse.urlparse(url)
- `core/agency/tool_orchestrator.py:214` [agency] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/api_adapter.py:105` [core_root] `aiohttp.ClientSession` - self._http_session = aiohttp.ClientSession(
- `core/api_adapter.py:106` [core_root] `aiohttp.TCPConnector` - connector=aiohttp.TCPConnector(limit=100, keepalive_timeout=60)
- `core/autonomic/iot_bridge.py:40` [autonomic] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/autonomic/iot_bridge.py:89` [autonomic] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/brain/react_loop.py:436` [brain] `httpx.get` - resp = httpx.get(search_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10.0, follow_redirects=True)
- `core/brain/react_loop.py:470` [brain] `httpx.get` - resp = httpx.get(target_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=15.0, follow_redirects=True)
- `core/bus/sensory_gate.py:209` [bus] `urllib.parse.quote` - f"&search={urllib.parse.quote(query)}&limit=3&namespace=0&format=json"
- `core/capabilities.py:65` [core_root] `asyncio.to_thread` - resp = await asyncio.to_thread(requests.get, url, headers=headers, timeout=self.timeout)
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
- `core/device_discovery.py:247` [core_root] `socket.socket` - with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
- `core/embodiment/iot_bridge.py:114` [embodiment] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/embodiment/mock_iot_plug.py:32` [embodiment] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/embodiment/mock_iot_plug.py:58` [embodiment] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/embodiment/mock_iot_plug.py:90` [embodiment] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:

## Degradation Handling

- Total `record_degradation()` calls: 2816
- Log-and-limp candidates: 2612
- Nearby fail-closed candidates: 204

Top limp-on files:

- `core/consciousness/consciousness_bridge.py`: 27
- `core/brain/inference_gate.py`: 26
- `core/resilience/memory_governor.py`: 25
- `core/runtime/runtime_hygiene.py`: 23
- `core/senses/voice_engine.py`: 23
- `core/memory/memory_facade.py`: 21
- `core/proactive_presence.py`: 21
- `core/self_modification/safe_modification.py`: 21
- `core/brain/llm/context_assembler.py`: 20
- `core/consciousness/liquid_substrate.py`: 19

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
- `core/narrative_thread.py`
- `core/reasoning/proof_answer_solver.py`
- `core/reproducibility/proof_substrate.py`
- `core/runtime/proof_kernel_bridge.py`
- `core/runtime/proof_policy.py`
- `core/search/research_pipeline.py`
- `core/skills/deep_research.py`

## Consolidation Candidates

- `core/audits/`: 2 file(s), 267 line(s)
- `core/body/`: 1 file(s), 139 line(s)
- `core/coherence/`: 2 file(s), 397 line(s)
- `core/consent/`: 2 file(s), 167 line(s)
- `core/constitution/`: 1 file(s), 25 line(s)
- `core/control/`: 2 file(s), 586 line(s)
- `core/creativity/`: 2 file(s), 801 line(s)
- `core/data/`: 2 file(s), 514 line(s)
- `core/ethics/`: 1 file(s), 310 line(s)
- `core/evals/`: 1 file(s), 143 line(s)
- `core/initializers/`: 2 file(s), 140 line(s)
- `core/intent/`: 1 file(s), 68 line(s)
- `core/latent/`: 1 file(s), 56 line(s)
- `core/llm/`: 2 file(s), 19 line(s)
- `core/maintenance/`: 2 file(s), 261 line(s)
- `core/media/`: 2 file(s), 273 line(s)
- `core/middleware/`: 2 file(s), 254 line(s)
- `core/multimodal/`: 2 file(s), 185 line(s)
- `core/networking/`: 1 file(s), 327 line(s)
- `core/ontology/`: 2 file(s), 169 line(s)
- `core/organism/`: 1 file(s), 476 line(s)
- `core/persistence/`: 2 file(s), 617 line(s)
- `core/planning/`: 1 file(s), 137 line(s)
- `core/play/`: 1 file(s), 228 line(s)
- `core/predictive/`: 2 file(s), 186 line(s)
- `core/reproducibility/`: 2 file(s), 497 line(s)
- `core/resource/`: 2 file(s), 430 line(s)
- `core/science/`: 1 file(s), 71 line(s)
- `core/search/`: 2 file(s), 1723 line(s)
- `core/sensors/`: 1 file(s), 151 line(s)
- `core/services/`: 2 file(s), 31 line(s)
- `core/session/`: 2 file(s), 231 line(s)
- `core/skill_management/`: 1 file(s), 367 line(s)
- `core/startup/`: 2 file(s), 326 line(s)
- `core/telemetry/`: 2 file(s), 191 line(s)
- `core/transparency/`: 2 file(s), 317 line(s)
- `core/twins/`: 1 file(s), 97 line(s)
- `core/values/`: 2 file(s), 289 line(s)
