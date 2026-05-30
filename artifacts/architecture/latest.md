# Aura Architecture Dependency Map

Schema: `aura.architecture.dependency_map.v1`
Root: `/Users/bryan/.aura/live-source`
Generated: `1780104318.812922`

## Summary

- Subsystems: 123
- Python files: 1589
- Python lines: 460928
- Dependency edges: 688
- ServiceContainer `.get()` calls: 1463
- ServiceContainer registrations: 350

## Subsystem Dependency Graph

```mermaid
graph TD
    runtime["runtime<br/>92 files, 20523 lines"]
    utils["utils<br/>41 files, 4956 lines"]
    brain["brain<br/>115 files, 42902 lines"]
    consciousness["consciousness<br/>121 files, 57071 lines"]
    resilience["resilience<br/>53 files, 11125 lines"]
    health["health<br/>3 files, 778 lines"]
    memory["memory<br/>70 files, 16745 lines"]
    agency["agency<br/>28 files, 14078 lines"]
    adaptation["adaptation<br/>26 files, 11869 lines"]
    constitution["constitution<br/>1 files, 25 lines"]
    senses["senses<br/>23 files, 5012 lines"]
    affect["affect<br/>5 files, 1857 lines"]
    self_modification["self_modification<br/>29 files, 10348 lines"]
    state["state<br/>6 files, 3408 lines"]
    observability["observability<br/>3 files, 575 lines"]
    governance["governance<br/>8 files, 2425 lines"]
    identity["identity<br/>11 files, 2136 lines"]
    orchestrator["orchestrator<br/>42 files, 18703 lines"]
    world_model["world_model<br/>9 files, 2541 lines"]
    executive["executive<br/>4 files, 2317 lines"]
    security["security<br/>17 files, 4623 lines"]
    tasks["tasks<br/>3 files, 333 lines"]
    autonomy["autonomy<br/>22 files, 7560 lines"]
    phases["phases<br/>29 files, 16970 lines"]
    actuators["actuators<br/>9 files, 2132 lines"]
    conversation["conversation<br/>8 files, 3993 lines"]
    learning["learning<br/>20 files, 6934 lines"]
    reasoning["reasoning<br/>7 files, 3830 lines"]
    skills["skills<br/>76 files, 15433 lines"]
    autonomic["autonomic<br/>4 files, 882 lines"]
    coordinators["coordinators<br/>9 files, 4224 lines"]
    managers["managers<br/>6 files, 932 lines"]
    meta["meta<br/>7 files, 1256 lines"]
    ops["ops<br/>11 files, 2262 lines"]
    organism["organism<br/>1 files, 458 lines"]
    self["self<br/>6 files, 1681 lines"]
    supervisor["supervisor<br/>3 files, 495 lines"]
    unity["unity<br/>11 files, 2409 lines"]
    agi["agi<br/>6 files, 1520 lines"]
    cognitive["cognitive<br/>11 files, 8097 lines"]
    collective["collective<br/>6 files, 1963 lines"]
    embodiment["embodiment<br/>15 files, 2627 lines"]
    ethics["ethics<br/>1 files, 309 lines"]
    evaluation["evaluation<br/>10 files, 1765 lines"]
    kernel["kernel<br/>10 files, 5397 lines"]
    motivation["motivation<br/>7 files, 1194 lines"]
    promotion["promotion<br/>6 files, 944 lines"]
    resource["resource<br/>2 files, 426 lines"]
    voice["voice<br/>7 files, 2776 lines"]
    world["world<br/>1 files, 297 lines"]
    cognition["cognition<br/>9 files, 3456 lines"]
    conversational["conversational<br/>4 files, 2237 lines"]
    data["data<br/>2 files, 514 lines"]
    db["db<br/>4 files, 583 lines"]
    goals["goals<br/>7 files, 3047 lines"]
    morphogenesis["morphogenesis<br/>12 files, 2854 lines"]
    pneuma["pneuma<br/>7 files, 1224 lines"]
    search["search<br/>2 files, 1715 lines"]
    somatic["somatic<br/>5 files, 2375 lines"]
    verification["verification<br/>4 files, 350 lines"]
    advanced_cognition["advanced_cognition<br/>13 files, 2905 lines"]
    architect["architect<br/>25 files, 5706 lines"]
    bus["bus<br/>4 files, 1748 lines"]
    coherence["coherence<br/>2 files, 397 lines"]
    discovery["discovery<br/>4 files, 579 lines"]
    environment["environment<br/>82 files, 8308 lines"]
    environments["environments<br/>7 files, 748 lines"]
    evolution["evolution<br/>6 files, 1855 lines"]
    introspection["introspection<br/>3 files, 738 lines"]
    lattice["lattice<br/>5 files, 704 lines"]
    maintenance["maintenance<br/>2 files, 261 lines"]
    perception["perception<br/>15 files, 3260 lines"]
    persistence["persistence<br/>2 files, 617 lines"]
    predictive["predictive<br/>2 files, 186 lines"]
    sandbox["sandbox<br/>4 files, 575 lines"]
    self_improvement["self_improvement<br/>12 files, 2285 lines"]
    sensors["sensors<br/>1 files, 151 lines"]
    services["services<br/>2 files, 31 lines"]
    simulation["simulation<br/>3 files, 390 lines"]
    soma["soma<br/>3 files, 502 lines"]
    sovereign["sovereign<br/>4 files, 541 lines"]
    startup["startup<br/>2 files, 326 lines"]
    workspace["workspace<br/>3 files, 1069 lines"]
    context["context<br/>4 files, 1199 lines"]
    creativity["creativity<br/>2 files, 800 lines"]
    curriculum["curriculum<br/>7 files, 657 lines"]
    cybernetics["cybernetics<br/>6 files, 1091 lines"]
    grounding["grounding<br/>7 files, 1092 lines"]
    guardians["guardians<br/>5 files, 623 lines"]
    llm["llm<br/>2 files, 19 lines"]
    media["media<br/>2 files, 273 lines"]
    middleware["middleware<br/>2 files, 254 lines"]
    networking["networking<br/>1 files, 327 lines"]
    plasticity["plasticity<br/>4 files, 342 lines"]
    research_core["research_core<br/>5 files, 580 lines"]
    safety["safety<br/>3 files, 628 lines"]
    skill_management["skill_management<br/>1 files, 350 lines"]
    social["social<br/>11 files, 3823 lines"]
    sovereignty["sovereignty<br/>3 files, 873 lines"]
    unknowns["unknowns<br/>4 files, 325 lines"]
    adapters["adapters<br/>3 files, 392 lines"]
    audits["audits<br/>2 files, 222 lines"]
    control["control<br/>2 files, 586 lines"]
    core_root["core_root<br/>182 files, 55004 lines"]
    distributed["distributed<br/>3 files, 140 lines"]
    initializers["initializers<br/>2 files, 140 lines"]
    intent["intent<br/>1 files, 68 lines"]
    knowledge["knowledge<br/>6 files, 142 lines"]
    latent["latent<br/>1 files, 56 lines"]
    multimodal["multimodal<br/>2 files, 176 lines"]
    neuroweb["neuroweb<br/>5 files, 368 lines"]
    ontology["ontology<br/>2 files, 169 lines"]
    pipeline["pipeline<br/>3 files, 217 lines"]
    planning["planning<br/>1 files, 137 lines"]
    play["play<br/>1 files, 228 lines"]
    providers["providers<br/>5 files, 857 lines"]
    reproducibility["reproducibility<br/>2 files, 497 lines"]
    session["session<br/>2 files, 225 lines"]
    systems["systems<br/>3 files, 256 lines"]
    telemetry["telemetry<br/>2 files, 191 lines"]
    temporal["temporal<br/>3 files, 1502 lines"]
    tools["tools<br/>2 files, 457 lines"]
    values["values<br/>2 files, 289 lines"]
    runtime --> actuators
    runtime --> adaptation
    runtime --> agency
    runtime --> architect
    runtime --> autonomy
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
    runtime --> self
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
    consciousness --> actuators
    consciousness --> adaptation
    consciousness --> affect
    consciousness --> agency
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
    health --> runtime
    health --> state
    memory --> actuators
    memory --> brain
    memory --> constitution
    memory --> db
    memory --> governance
    memory --> health
    memory --> phases
    memory --> resilience
    memory --> runtime
    memory --> utils
    agency --> adaptation
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
    adaptation --> resilience
    adaptation --> runtime
    adaptation --> sensors
    adaptation --> utils
    adaptation --> world
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
    affect --> adaptation
    affect --> autonomic
    affect --> brain
    affect --> consciousness
    affect --> health
    affect --> memory
    affect --> runtime
    affect --> senses
    affect --> utils
    self_modification --> ethics
    self_modification --> governance
    self_modification --> resilience
    self_modification --> runtime
    self_modification --> skills
    self_modification --> utils
    state --> constitution
    state --> governance
    state --> motivation
    state --> runtime
    state --> unity
    state --> utils
    observability --> runtime
    governance --> actuators
    governance --> consciousness
    governance --> runtime
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
    executive --> runtime
    executive --> state
    security --> affect
    security --> agency
    security --> consciousness
    security --> identity
    security --> memory
    security --> runtime
    security --> utils
    tasks --> runtime
    autonomy --> affect
    autonomy --> agency
    autonomy --> consciousness
    autonomy --> executive
    autonomy --> observability
    autonomy --> resource
    autonomy --> runtime
    autonomy --> state
    autonomy --> utils
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
    actuators --> affect
    actuators --> brain
    actuators --> executive
    actuators --> memory
    actuators --> runtime
    actuators --> sandbox
    actuators --> search
    actuators --> skills
    actuators --> world
    conversation --> brain
    conversation --> consciousness
    conversation --> memory
    conversation --> runtime
    conversation --> utils
    learning --> brain
    learning --> consciousness
    learning --> introspection
    learning --> promotion
    learning --> reasoning
    learning --> runtime
    learning --> self_modification
    learning --> skills
    learning --> tasks
    learning --> utils
    reasoning --> runtime
    skills --> actuators
    skills --> advanced_cognition
    skills --> affect
    skills --> brain
    skills --> embodiment
    skills --> executive
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
    kernel --> orchestrator
    kernel --> phases
    kernel --> resilience
    kernel --> runtime
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
    voice --> resilience
    voice --> runtime
    voice --> senses
    voice --> utils
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
    search --> runtime
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
    evolution --> autonomy
    evolution --> brain
    evolution --> runtime
    evolution --> utils
    introspection --> runtime
    maintenance --> resilience
    maintenance --> runtime
    perception --> brain
    persistence --> observability
    persistence --> resilience
    persistence --> runtime
    predictive --> brain
    predictive --> runtime
    predictive --> utils
    sandbox --> runtime
    self_improvement --> brain
    self_improvement --> runtime
    self_improvement --> self_modification
    sensors --> world
    services --> autonomic
    simulation --> brain
    simulation --> consciousness
    simulation --> runtime
    simulation --> world_model
    soma --> resilience
    soma --> runtime
    soma --> utils
    sovereign --> runtime
    startup --> brain
    startup --> runtime
    startup --> senses
    workspace --> runtime
    context --> runtime
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
    social --> agency
    social --> ethics
    social --> governance
    social --> runtime
    social --> utils
    sovereignty --> ethics
    sovereignty --> governance
    sovereignty --> identity
    sovereignty --> organism
    sovereignty --> runtime
    sovereignty --> utils
    unknowns --> lattice
    unknowns --> promotion
    unknowns --> verification
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
    core_root --> utils
    core_root --> voice
    core_root --> workspace
    core_root --> world_model
    initializers --> adaptation
    initializers --> consciousness
    initializers --> introspection
    initializers --> memory
    initializers --> meta
    initializers --> runtime
    initializers --> senses
    initializers --> utils
    intent --> runtime
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
    providers --> world_model
    reproducibility --> runtime
    session --> runtime
    systems --> runtime
    systems --> services
    temporal --> runtime
    temporal --> utils
    tools --> runtime
    tools --> skills
```

## Core Subsystem Stats

| Subsystem | Files | Lines | Bytes | Deps Out | Deps In |
| --- | ---: | ---: | ---: | ---: | ---: |
| consciousness | 121 | 57071 | 2403751 | 37 | 29 |
| core_root | 182 | 55004 | 2237533 | 99 | 0 |
| brain | 115 | 42902 | 1853761 | 40 | 39 |
| runtime | 92 | 20523 | 723797 | 38 | 104 |
| orchestrator | 42 | 18703 | 826005 | 123 | 9 |
| phases | 29 | 16970 | 764881 | 32 | 7 |
| memory | 70 | 16745 | 677304 | 17 | 21 |
| skills | 76 | 15433 | 622714 | 27 | 6 |
| agency | 28 | 14078 | 569142 | 27 | 17 |
| adaptation | 26 | 11869 | 475074 | 18 | 14 |
| resilience | 53 | 11125 | 448837 | 16 | 25 |
| self_modification | 29 | 10348 | 409314 | 11 | 13 |
| environment | 82 | 8308 | 322051 | 10 | 2 |
| cognitive | 11 | 8097 | 329816 | 9 | 4 |
| autonomy | 22 | 7560 | 310198 | 16 | 7 |
| learning | 20 | 6934 | 274949 | 14 | 6 |
| architect | 25 | 5706 | 238326 | 9 | 2 |
| kernel | 10 | 5397 | 222452 | 19 | 4 |
| senses | 23 | 5012 | 210643 | 16 | 14 |
| utils | 41 | 4956 | 192837 | 17 | 49 |
| security | 17 | 4623 | 182059 | 11 | 8 |
| coordinators | 9 | 4224 | 198441 | 36 | 5 |
| conversation | 8 | 3993 | 147860 | 10 | 6 |
| reasoning | 7 | 3830 | 155566 | 2 | 6 |
| social | 11 | 3823 | 163226 | 8 | 1 |
| cognition | 9 | 3456 | 139831 | 7 | 3 |
| state | 6 | 3408 | 144522 | 8 | 13 |
| perception | 15 | 3260 | 129879 | 3 | 2 |
| goals | 7 | 3047 | 129238 | 6 | 3 |
| advanced_cognition | 13 | 2905 | 118305 | 3 | 2 |
| morphogenesis | 12 | 2854 | 111137 | 8 | 3 |
| voice | 7 | 2776 | 127072 | 9 | 4 |
| embodiment | 15 | 2627 | 102452 | 12 | 4 |
| world_model | 9 | 2541 | 103649 | 8 | 9 |
| governance | 8 | 2425 | 98518 | 8 | 10 |
| unity | 11 | 2409 | 100598 | 3 | 5 |
| somatic | 5 | 2375 | 89958 | 7 | 3 |
| executive | 4 | 2317 | 97890 | 13 | 8 |
| self_improvement | 12 | 2285 | 86875 | 3 | 2 |
| ops | 11 | 2262 | 87136 | 15 | 5 |
| conversational | 4 | 2237 | 95525 | 4 | 3 |
| identity | 11 | 2136 | 90455 | 9 | 9 |
| actuators | 9 | 2132 | 83552 | 12 | 6 |
| collective | 6 | 1963 | 81012 | 8 | 4 |
| affect | 5 | 1857 | 84809 | 12 | 13 |
| evolution | 6 | 1855 | 75408 | 8 | 2 |
| evaluation | 10 | 1765 | 61879 | 3 | 4 |
| bus | 4 | 1748 | 73318 | 6 | 2 |
| search | 2 | 1715 | 64882 | 6 | 3 |
| self | 6 | 1681 | 70508 | 12 | 5 |
| agi | 6 | 1520 | 63343 | 13 | 4 |
| temporal | 3 | 1502 | 51551 | 2 | 0 |
| meta | 7 | 1256 | 47229 | 5 | 5 |
| pneuma | 7 | 1224 | 46166 | 4 | 3 |
| context | 4 | 1199 | 46480 | 1 | 1 |
| motivation | 7 | 1194 | 50455 | 10 | 4 |
| grounding | 7 | 1092 | 40453 | 4 | 1 |
| cybernetics | 6 | 1091 | 43608 | 6 | 1 |
| workspace | 3 | 1069 | 39594 | 3 | 2 |
| promotion | 6 | 944 | 31711 | 1 | 4 |
| managers | 6 | 932 | 39695 | 25 | 5 |
| autonomic | 4 | 882 | 36599 | 5 | 5 |
| sovereignty | 3 | 873 | 33628 | 10 | 1 |
| providers | 5 | 857 | 40239 | 51 | 0 |
| creativity | 2 | 800 | 33218 | 2 | 1 |
| health | 3 | 778 | 29160 | 6 | 21 |
| environments | 7 | 748 | 31101 | 3 | 2 |
| introspection | 3 | 738 | 28467 | 1 | 2 |
| lattice | 5 | 704 | 26089 | 0 | 2 |
| curriculum | 7 | 657 | 21995 | 1 | 1 |
| safety | 3 | 628 | 25730 | 3 | 1 |
| guardians | 5 | 623 | 27309 | 6 | 1 |
| persistence | 2 | 617 | 24953 | 3 | 2 |
| control | 2 | 586 | 20961 | 4 | 0 |
| db | 4 | 583 | 22581 | 2 | 3 |
| research_core | 5 | 580 | 22543 | 8 | 1 |
| discovery | 4 | 579 | 20581 | 2 | 2 |
| observability | 3 | 575 | 20634 | 4 | 11 |
| sandbox | 4 | 575 | 20176 | 1 | 2 |
| sovereign | 4 | 541 | 18838 | 1 | 2 |
| data | 2 | 514 | 19319 | 2 | 3 |
| soma | 3 | 502 | 19729 | 3 | 2 |
| reproducibility | 2 | 497 | 18141 | 1 | 0 |
| supervisor | 3 | 495 | 17963 | 1 | 5 |
| organism | 1 | 458 | 17881 | 3 | 5 |
| tools | 2 | 457 | 17582 | 2 | 0 |
| resource | 2 | 426 | 15523 | 3 | 4 |
| coherence | 2 | 397 | 18920 | 6 | 2 |
| adapters | 3 | 392 | 13257 | 0 | 0 |
| simulation | 3 | 390 | 15477 | 6 | 2 |
| neuroweb | 5 | 368 | 14195 | 5 | 0 |
| skill_management | 1 | 350 | 17043 | 6 | 1 |
| verification | 4 | 350 | 13177 | 2 | 3 |
| plasticity | 4 | 342 | 12056 | 2 | 1 |
| tasks | 3 | 333 | 11388 | 3 | 8 |
| networking | 1 | 327 | 12150 | 3 | 1 |
| startup | 2 | 326 | 11280 | 5 | 2 |
| unknowns | 4 | 325 | 11829 | 3 | 1 |
| ethics | 1 | 309 | 11902 | 1 | 4 |
| world | 1 | 297 | 10842 | 0 | 4 |
| values | 2 | 289 | 10861 | 0 | 0 |
| media | 2 | 273 | 9349 | 0 | 1 |
| maintenance | 2 | 261 | 9351 | 4 | 2 |
| systems | 3 | 256 | 9861 | 3 | 0 |
| middleware | 2 | 254 | 10813 | 2 | 1 |
| play | 1 | 228 | 8774 | 4 | 0 |
| session | 2 | 225 | 9184 | 1 | 0 |
| audits | 2 | 222 | 8524 | 2 | 0 |
| pipeline | 3 | 217 | 6684 | 1 | 0 |
| telemetry | 2 | 191 | 5594 | 0 | 0 |
| predictive | 2 | 186 | 7105 | 5 | 2 |
| multimodal | 2 | 176 | 6358 | 0 | 0 |
| ontology | 2 | 169 | 5381 | 0 | 0 |
| sensors | 1 | 151 | 5644 | 1 | 2 |
| knowledge | 6 | 142 | 3870 | 0 | 0 |
| distributed | 3 | 140 | 4655 | 0 | 0 |
| initializers | 2 | 140 | 6565 | 10 | 0 |
| planning | 1 | 137 | 5440 | 2 | 0 |
| intent | 1 | 68 | 2661 | 1 | 0 |
| latent | 1 | 56 | 2337 | 0 | 0 |
| services | 2 | 31 | 1171 | 1 | 2 |
| constitution | 1 | 25 | 795 | 0 | 14 |
| llm | 2 | 19 | 745 | 1 | 1 |

## ServiceContainer Cross-Wiring

- Unique services retrieved: 359
- Unique services registered: 287
- Services retrieved without detected registration: 184

### Top Fetched Services

| Service | Gets | Registrations |
| --- | ---: | ---: |
| orchestrator | 67 | 3 |
| cognitive_engine | 52 | 3 |
| llm_router | 40 | 2 |
| affect_engine | 38 | 1 |
| inference_gate | 37 | 4 |
| capability_engine | 31 | 2 |
| memory_facade | 29 | 1 |
| liquid_substrate | 28 | 1 |
| mycelial_network | 28 | 2 |
| free_energy_engine | 24 | 0 |
| homeostasis | 22 | 1 |
| global_workspace | 22 | 2 |
| drive_engine | 22 | 0 |
| goal_engine | 20 | 0 |
| state_repository | 19 | 1 |
| knowledge_graph | 18 | 0 |
| qualia_synthesizer | 18 | 3 |
| belief_revision_engine | 17 | 1 |
| episodic_memory | 17 | 1 |
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
- `api_adapter` fetched 5 time(s)
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
| UnifiedWill decisions | 54 | 28 | 2 | 52 |
| Memory writes | 267 | 104 | 45 | 222 |
| State mutation | 373 | 142 | 5 | 368 |
| Tool execution | 136 | 75 | 3 | 133 |
| Self-modification and patching | 15 | 12 | 2 | 13 |
| LLM inference | 245 | 147 | 65 | 180 |
| External I/O | 165 | 82 | 15 | 150 |

### UnifiedWill decisions

Calls that can ask the single will authority to approve action.

Review candidates:
- `core/actuators/actuator_synthesis.py:183` [actuators] `get_will` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:183` [actuators] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/adaptive_immunity.py:1986` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/adaptive_immunity.py:1986` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:624` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:624` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:168` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:168` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/agency/agency_bus.py:83` [agency] `get_will` - _auto_decision = get_will().decide(
- `core/agency/agency_bus.py:83` [agency] `get_will.decide` - _auto_decision = get_will().decide(
- `core/autonomy/self_modification.py:272` [autonomy] `will.decide` - decision = will.decide(
- `core/cognitive/autopoiesis.py:961` [cognitive] `will.decide` - decision = will.decide(
- `core/consciousness/parallel_branches.py:736` [consciousness] `will.decide` - decision = will.decide(
- `core/environment/governance_bridge.py:47` [environment] `self.will_gateway.decide` - will_decision = await self.will_gateway.decide(intent)
- `core/goals/goal_engine.py:1013` [goals] `will.decide` - decision = will.decide(
- `core/governance/will_gate.py:109` [governance] `will.decide` - decision = will.decide(
- `core/governance/will_gate.py:160` [governance] `will.decide` - decision = will.decide(
- `core/initiative_synthesis.py:743` [core_root] `get_will` - decision = get_will().decide(
- `core/initiative_synthesis.py:743` [core_root] `get_will.decide` - decision = get_will().decide(
- `core/learning/genuine_learning_pipeline.py:643` [learning] `get_will` - decision = get_will().decide(
- `core/learning/genuine_learning_pipeline.py:643` [learning] `get_will.decide` - decision = get_will().decide(
- `core/learning/recursive_self_improvement.py:502` [learning] `get_will` - decision = get_will().decide(
- `core/learning/recursive_self_improvement.py:502` [learning] `get_will.decide` - decision = get_will().decide(
- `core/mind_tick.py:104` [core_root] `get_will` - decision = get_will().decide(
- `core/mind_tick.py:104` [core_root] `get_will.decide` - decision = get_will().decide(

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
- `core/autonomy/research_cycle.py:706` [autonomy] `memory_facade.add_memory` - result = memory_facade.add_memory(memory_payload, metadata=metadata)

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
- `core/adaptation/meta_learner.py:296` [adaptation] `np.savez_compressed` - np.savez_compressed(str(_STATE_PATH), **save_dict)
- `core/adaptation/value_autopoiesis.py:140` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:232` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:289` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:510` [adaptation] `os.replace` - os.replace(tmp_path, _STATE_PATH)
- `core/advanced_cognition/integration.py:132` [advanced_cognition] `next_state.setdefault` - next_state.setdefault("_advanced_prediction", {})[act.action_id] = pred
- `core/advanced_cognition/integration.py:223` [advanced_cognition] `issubset` - if isinstance(value, Mapping) and {"domain", "state"}.issubset(value.keys()):
- `core/advanced_cognition/ontology_invention.py:156` [advanced_cognition] `self.save` - self.save(self.state_path)
- `core/advanced_cognition/world_model.py:73` [advanced_cognition] `self.save` - self.save(self.state_path)
- `core/advanced_cognition/zero_shot_transfer.py:75` [advanced_cognition] `self.save` - self.save(self.state_path)
- `core/agency/agency_core.py:132` [agency] `get_registry.update` - get_registry().update(active_shards=len(self.active_shards)),
- `core/agency/agency_core.py:778` [agency] `virtual_body.__dict__.update` - virtual_body.__dict__.update(snapshot)
- `core/agency/agency_core.py:964` [agency] `get_registry.update` - get_registry().update(
- `core/agency/autonomous_task_engine.py:425` [agency] `self._update_state_goals` - self._update_state_goals(plan)
- `core/agency/autonomous_task_engine.py:661` [agency] `self._update_state_goals` - self._update_state_goals(plan)
- `core/agency/autonomous_task_engine.py:681` [agency] `self._update_state_goals` - self._update_state_goals(plan)
- `core/agency/autonomous_task_engine.py:767` [agency] `self._update_state_goals` - self._update_state_goals(plan)
- `core/architect/lesion_matrix.py:135` [architect] `self._set_state` - self._set_state(self._saved_state)

### Tool execution

Calls that can execute tools, skills, shells, browsers, or external actions.

Review candidates:
- `core/actuators/actuator_registry.py:375` [actuators] `self.operator.execute_synthesized_tool` - res = self.operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/code_execution_actuator.py:98` [actuators] `operator.execute_synthesized_tool` - res = operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/web_actuators.py:114` [actuators] `skill.execute` - return await skill.execute({"mode": "browse", "url": url}, {})
- `core/agency/agency_core.py:410` [agency] `self._execute_shard_tool` - tasks.append(self._execute_shard_tool(name, payload))
- `core/agency/agency_orchestrator.py:360` [agency] `execute` - await execute(proposal, state_snapshot, receipt.capability_token or "")
- `core/agency/autonomous_task_engine.py:515` [agency] `orchestrator.execute_tool` - return await orchestrator.execute_tool(tool_name, args, **kwargs)
- `core/agency/autonomous_task_engine.py:2715` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:2718` [agency] `orch.execute_tool` - return await orch.execute_tool("web_search", {"query": query})
- `core/agency/autonomous_task_engine.py:2733` [agency] `orch.execute_tool` - result = await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:2737` [agency] `orch.execute_tool` - result = await orch.execute_tool("run_python", {"code": code})
- `core/agency/skill_library.py:199` [agency] `tool_orchestrator.execute_tool` - result = await tool_orchestrator.execute_tool(step.tool_name, resolved_args)
- `core/agi/curiosity_explorer.py:240` [agi] `orchestrator.execute_tool` - orchestrator.execute_tool(
- `core/architect/safety_gate.py:304` [architect] `subprocess.run` - result = subprocess.run(
- `core/architect/safety_gate.py:359` [architect] `subprocess.run` - subprocess.run(
- `core/architect/safety_gate.py:365` [architect] `subprocess.run` - result = subprocess.run(
- `core/architect/safety_gate.py:379` [architect] `subprocess.run` - snap["head"] = subprocess.run(
- `core/architect/safety_gate.py:384` [architect] `subprocess.run` - snap["status"] = subprocess.run(
- `core/architect/safety_gate.py:389` [architect] `subprocess.run` - snap["diff_stat"] = subprocess.run(
- `core/architect/shadow_workspace.py:115` [architect] `subprocess.run` - proc = subprocess.run(
- `core/architect/shadow_workspace.py:204` [architect] `subprocess.run` - proc = subprocess.run(
- `core/autonomous_initiative_loop.py:473` [core_root] `capability_engine.execute` - scan_result = await capability_engine.execute(
- `core/autonomous_initiative_loop.py:519` [core_root] `capability_engine.execute` - test_result = await capability_engine.execute(
- `core/autonomous_initiative_loop.py:558` [core_root] `capability_engine.execute` - proposal_result = await capability_engine.execute(
- `core/autonomous_initiative_loop.py:932` [core_root] `skill.execute` - return await skill.execute(EmailInput(**payload), {})
- `core/autonomous_initiative_loop.py:944` [core_root] `skill.execute` - return await skill.execute(RedditInput(**payload), {})

### Self-modification and patching

Calls that can generate, validate, apply, or promote code changes.

Review candidates:
- `core/architect/governor.py:140` [architect] `self.promotion_governor.promote` - decision = self.promotion_governor.promote(plan, shadow, proof, rollback)
- `core/brain/cognitive_patch.py:91` [brain] `f.write` - f.write(f"# Cognitive patch proposal — REQUIRES MANUAL REVIEW\n")
- `core/guardians/airlock.py:80` [guardians] `atomic_write_text` - atomic_write_text(patch_file, diff_patch, encoding="utf-8")
- `core/kernel/upgrades_10x.py:335` [kernel] `self._safe_self_modify` - await self._safe_self_modify(state)
- `core/optimizer.py:59` [core_root] `patch.apply` - success = await patch.apply(signature)
- `core/optimizer.py:61` [core_root] `patch.apply` - success = await patch.apply()
- `core/optimizer.py:72` [core_root] `cog_patch.apply` - if await cog_patch.apply(signature):
- `core/orchestrator/mixins/boot/boot_autonomy.py:853` [orchestrator] `apply_presence_patch` - apply_presence_patch(self)
- `core/skill_management/hephaestus.py:194` [skill_management] `guard.validate` - if not guard.validate(patched_code):
- `core/skills/self_repair.py:122` [skills] `atomic_write_text` - atomic_write_text(patch_path, fix_content)
- `core/state/cellular_substrate.py:64` [state] `self._apply_patch_recursive` - self._apply_patch_recursive(state, patch)
- `core/state/cellular_substrate.py:82` [state] `self._apply_patch_recursive` - self._apply_patch_recursive(sub_target, value)
- `core/utils/sandbox_selfmod.py:60` [utils] `fh.write` - fh.write(patch_text)

### LLM inference

Calls that can spend model context or produce model-authored text/code.

Review candidates:
- `core/actuators/actuator_synthesis.py:157` [actuators] `brain.generate` - res = await brain.generate(prompt, system_prompt=system_prompt)
- `core/adaptation/distillation_pipe.py:103` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/distillation_pipe.py:145` [adaptation] `router.think` - response = await router.think(
- `core/adaptation/dream_journal.py:161` [adaptation] `self.brain.think` - res = await self.brain.think(
- `core/adaptation/epistemic_humility.py:145` [adaptation] `llm.chat` - response = await llm.chat(
- `core/adaptation/heuristic_synthesizer.py:126` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/star_reasoner.py:364` [adaptation] `llm.think` - result = await asyncio.wait_for(llm.think(prompt), timeout=self.RATIONALIZATION_TIMEOUT)
- `core/agency/agency_core.py:290` [agency] `structured_brain.generate` - shard_res = await structured_brain.generate(prompt, context=context)
- `core/agency/autonomous_task_engine.py:962` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2467` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2501` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2585` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2694` [agency] `llm.think` - raw = await llm.think(
- `core/agency/latent_distiller.py:49` [agency] `brain.think` - summary = await brain.think(
- `core/agi/curiosity_explorer.py:310` [agi] `router.think` - router.think(prompt, priority=0.3, is_background=True,
- `core/agi/hierarchical_planner.py:215` [agi] `router.think` - router.think(prompt, priority=0.3, is_background=True,
- `core/agi/skill_synthesizer.py:174` [agi] `router.think` - router.think(prompt, priority=0.2, is_background=True,
- `core/audits/alignment_auditor.py:44` [audits] `self.brain.think` - response = await self.brain.think(
- `core/audits/alignment_auditor.py:99` [audits] `self.brain.think` - response = await self.brain.think(
- `core/audits/tool_auditor.py:34` [audits] `self.brain.think` - thought = await self.brain.think(
- `core/autonomy/genuine_refusal.py:334` [autonomy] `llm.think` - llm.think(prompt, mode="FAST"),
- `core/autonomy/genuine_refusal.py:376` [autonomy] `llm.think` - llm.think(prompt, mode="FAST"),
- `core/autonomy/genuine_refusal.py:403` [autonomy] `llm.think` - llm.think(prompt, mode="FAST"),
- `core/autonomy/personhood_engine.py:187` [autonomy] `llm.think` - llm.think(f"[Spontaneous Thought Prompt] {prompt}", mode="FAST"),
- `core/autonomy/research_cycle.py:579` [autonomy] `llm.think` - return await llm.think(prompt)

### External I/O

Calls that can touch network, subprocesses, sockets, browsers, or APIs.

Review candidates:
- `core/actuators/web_actuators.py:88` [actuators] `urllib.parse.urlparse` - parsed = urllib.parse.urlparse(url)
- `core/agency/tool_orchestrator.py:212` [agency] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/api_adapter.py:105` [core_root] `aiohttp.ClientSession` - self._http_session = aiohttp.ClientSession(
- `core/api_adapter.py:106` [core_root] `aiohttp.TCPConnector` - connector=aiohttp.TCPConnector(limit=100, keepalive_timeout=60)
- `core/architect/safety_gate.py:304` [architect] `subprocess.run` - result = subprocess.run(
- `core/architect/safety_gate.py:359` [architect] `subprocess.run` - subprocess.run(
- `core/architect/safety_gate.py:365` [architect] `subprocess.run` - result = subprocess.run(
- `core/architect/safety_gate.py:379` [architect] `subprocess.run` - snap["head"] = subprocess.run(
- `core/architect/safety_gate.py:384` [architect] `subprocess.run` - snap["status"] = subprocess.run(
- `core/architect/safety_gate.py:389` [architect] `subprocess.run` - snap["diff_stat"] = subprocess.run(
- `core/architect/shadow_workspace.py:115` [architect] `subprocess.run` - proc = subprocess.run(
- `core/architect/shadow_workspace.py:204` [architect] `subprocess.run` - proc = subprocess.run(
- `core/autonomic/iot_bridge.py:40` [autonomic] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/autonomic/iot_bridge.py:89` [autonomic] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/autonomy/content_fetcher.py:584` [autonomy] `urllib.request.Request` - req = urllib.request.Request(url, headers={"User-Agent": "Aura/1.0 (+research)"})
- `core/autonomy/content_fetcher.py:587` [autonomy] `urllib.request.urlopen` - with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
- `core/brain/llm/gemini_adapter.py:316` [brain] `httpx.AsyncClient` - self._client = httpx.AsyncClient(
- `core/brain/llm/gemini_adapter.py:317` [brain] `httpx.Timeout` - timeout=httpx.Timeout(self.timeout, connect=10.0),
- `core/brain/llm/llm_router.py:327` [brain] `httpx.AsyncClient` - async with httpx.AsyncClient(timeout=self.endpoint.timeout) as client:
- `core/brain/llm/local_llm_setup.py:78` [brain] `httpx.AsyncClient` - async with httpx.AsyncClient() as client:
- `core/brain/llm/local_llm_setup.py:101` [brain] `subprocess.run` - subprocess.run(
- `core/brain/llm/local_llm_setup.py:121` [brain] `subprocess.run` - res = subprocess.run(
- `core/brain/llm/local_llm_setup.py:130` [brain] `subprocess.run` - subprocess.run(
- `core/brain/llm/local_server_client.py:626` [brain] `urllib.request.Request` - req = urllib.request.Request(url, method="GET")
- `core/brain/llm/local_server_client.py:627` [brain] `urllib.request.urlopen` - with urllib.request.urlopen(req, timeout=2.0) as resp:

## Degradation Handling

- Total `record_degradation()` calls: 2678
- Log-and-limp candidates: 2483
- Nearby fail-closed candidates: 195

Top limp-on files:

- `core/consciousness/consciousness_bridge.py`: 27
- `core/brain/inference_gate.py`: 26
- `core/resilience/memory_governor.py`: 25
- `core/senses/voice_engine.py`: 23
- `core/proactive_presence.py`: 21
- `core/self_modification/safe_modification.py`: 21
- `core/brain/llm/context_assembler.py`: 19
- `core/consciousness/liquid_substrate.py`: 19
- `core/memory/memory_facade.py`: 19
- `core/agency/autonomous_task_engine.py`: 18

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
- `core/learning/proof_obligations.py`
- `core/narrative_thread.py`
- `core/reasoning/proof_answer_solver.py`
- `core/reproducibility/proof_substrate.py`
- `core/runtime/proof_kernel_bridge.py`
- `core/runtime/proof_policy.py`
- `core/search/research_pipeline.py`
- `core/skills/deep_research.py`

## Consolidation Candidates

- `core/audits/`: 2 file(s), 222 line(s)
- `core/coherence/`: 2 file(s), 397 line(s)
- `core/constitution/`: 1 file(s), 25 line(s)
- `core/control/`: 2 file(s), 586 line(s)
- `core/creativity/`: 2 file(s), 800 line(s)
- `core/data/`: 2 file(s), 514 line(s)
- `core/ethics/`: 1 file(s), 309 line(s)
- `core/initializers/`: 2 file(s), 140 line(s)
- `core/intent/`: 1 file(s), 68 line(s)
- `core/latent/`: 1 file(s), 56 line(s)
- `core/llm/`: 2 file(s), 19 line(s)
- `core/maintenance/`: 2 file(s), 261 line(s)
- `core/media/`: 2 file(s), 273 line(s)
- `core/middleware/`: 2 file(s), 254 line(s)
- `core/multimodal/`: 2 file(s), 176 line(s)
- `core/networking/`: 1 file(s), 327 line(s)
- `core/ontology/`: 2 file(s), 169 line(s)
- `core/organism/`: 1 file(s), 458 line(s)
- `core/persistence/`: 2 file(s), 617 line(s)
- `core/planning/`: 1 file(s), 137 line(s)
- `core/play/`: 1 file(s), 228 line(s)
- `core/predictive/`: 2 file(s), 186 line(s)
- `core/reproducibility/`: 2 file(s), 497 line(s)
- `core/resource/`: 2 file(s), 426 line(s)
- `core/search/`: 2 file(s), 1715 line(s)
- `core/sensors/`: 1 file(s), 151 line(s)
- `core/services/`: 2 file(s), 31 line(s)
- `core/session/`: 2 file(s), 225 line(s)
- `core/skill_management/`: 1 file(s), 350 line(s)
- `core/startup/`: 2 file(s), 326 line(s)
- `core/telemetry/`: 2 file(s), 191 line(s)
- `core/tools/`: 2 file(s), 457 line(s)
- `core/values/`: 2 file(s), 289 line(s)
- `core/world/`: 1 file(s), 297 line(s)
