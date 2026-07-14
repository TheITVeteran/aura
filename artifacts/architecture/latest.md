# Aura Architecture Dependency Map

Schema: `aura.architecture.dependency_map.v2`
Root: `<AURA_ROOT>`
Generated: `0.0`

## Summary

- Subsystems: 148
- Python files: 2177
- Python lines: 661422
- Dependency edges: 1071
- ServiceContainer `.get()` calls: 1439
- ServiceContainer registrations: 331
- Boot contract: PASS

## Subsystem Dependency Graph

```mermaid
graph TD
    runtime["runtime<br/>146 files, 47478 lines"]
    utils["utils<br/>45 files, 6745 lines"]
    brain["brain<br/>162 files, 64203 lines"]
    memory["memory<br/>94 files, 23556 lines"]
    consciousness["consciousness<br/>149 files, 68953 lines"]
    resilience["resilience<br/>66 files, 16902 lines"]
    health["health<br/>5 files, 1121 lines"]
    agency["agency<br/>49 files, 18870 lines"]
    observability["observability<br/>11 files, 2583 lines"]
    governance["governance<br/>10 files, 3593 lines"]
    adaptation["adaptation<br/>28 files, 13232 lines"]
    affect["affect<br/>12 files, 4161 lines"]
    constitution["constitution<br/>1 files, 25 lines"]
    security["security<br/>39 files, 9009 lines"]
    senses["senses<br/>27 files, 7004 lines"]
    identity["identity<br/>18 files, 2734 lines"]
    self_modification["self_modification<br/>35 files, 12733 lines"]
    conversation["conversation<br/>19 files, 11441 lines"]
    cognition["cognition<br/>24 files, 8641 lines"]
    state["state<br/>9 files, 4440 lines"]
    perception["perception<br/>29 files, 11205 lines"]
    orchestrator["orchestrator<br/>45 files, 21144 lines"]
    skills["skills<br/>94 files, 28365 lines"]
    world_model["world_model<br/>11 files, 3290 lines"]
    autonomy["autonomy<br/>29 files, 11861 lines"]
    epistemics["epistemics<br/>14 files, 4834 lines"]
    executive["executive<br/>11 files, 3278 lines"]
    social["social<br/>21 files, 7779 lines"]
    continuity["continuity<br/>7 files, 238 lines"]
    organism["organism<br/>9 files, 1949 lines"]
    values["values<br/>15 files, 1953 lines"]
    being["being<br/>28 files, 6931 lines"]
    learning["learning<br/>41 files, 13244 lines"]
    knowledge["knowledge<br/>10 files, 1112 lines"]
    morality["morality<br/>16 files, 1327 lines"]
    tasks["tasks<br/>5 files, 597 lines"]
    actuators["actuators<br/>10 files, 2333 lines"]
    capabilities["capabilities<br/>20 files, 11426 lines"]
    introspection["introspection<br/>8 files, 2164 lines"]
    phases["phases<br/>29 files, 20096 lines"]
    planning["planning<br/>9 files, 4035 lines"]
    reasoning["reasoning<br/>12 files, 5307 lines"]
    self["self<br/>7 files, 2196 lines"]
    voice["voice<br/>10 files, 4427 lines"]
    world["world<br/>24 files, 1483 lines"]
    bus["bus<br/>6 files, 2545 lines"]
    discovery["discovery<br/>7 files, 2134 lines"]
    kernel["kernel<br/>11 files, 6530 lines"]
    ops["ops<br/>18 files, 5377 lines"]
    autonomic["autonomic<br/>5 files, 1237 lines"]
    cognitive["cognitive<br/>12 files, 9234 lines"]
    coordinators["coordinators<br/>10 files, 4604 lines"]
    embodiment["embodiment<br/>15 files, 3139 lines"]
    ethics["ethics<br/>2 files, 580 lines"]
    managers["managers<br/>6 files, 959 lines"]
    meta["meta<br/>7 files, 1276 lines"]
    resource["resource<br/>4 files, 624 lines"]
    sleep["sleep<br/>9 files, 858 lines"]
    supervisor["supervisor<br/>3 files, 976 lines"]
    unity["unity<br/>11 files, 2791 lines"]
    agi["agi<br/>6 files, 1562 lines"]
    collective["collective<br/>6 files, 2079 lines"]
    data["data<br/>3 files, 651 lines"]
    evaluation["evaluation<br/>16 files, 2983 lines"]
    goals["goals<br/>10 files, 3535 lines"]
    media["media<br/>2 files, 336 lines"]
    motivation["motivation<br/>7 files, 1209 lines"]
    phenomenal_substrate["phenomenal_substrate<br/>11 files, 1042 lines"]
    promotion["promotion<br/>6 files, 936 lines"]
    sandbox["sandbox<br/>4 files, 610 lines"]
    self_improvement["self_improvement<br/>14 files, 5586 lines"]
    somatic["somatic<br/>6 files, 2894 lines"]
    adapters["adapters<br/>5 files, 1018 lines"]
    advanced_cognition["advanced_cognition<br/>13 files, 2905 lines"]
    conversational["conversational<br/>4 files, 2434 lines"]
    db["db<br/>4 files, 584 lines"]
    environment["environment<br/>83 files, 9202 lines"]
    ghost["ghost<br/>6 files, 1961 lines"]
    llm["llm<br/>3 files, 200 lines"]
    morphogenesis["morphogenesis<br/>12 files, 2875 lines"]
    pneuma["pneuma<br/>7 files, 1257 lines"]
    search["search<br/>2 files, 1758 lines"]
    verification["verification<br/>4 files, 350 lines"]
    workspace["workspace<br/>9 files, 1242 lines"]
    architect["architect<br/>25 files, 5743 lines"]
    coherence["coherence<br/>2 files, 407 lines"]
    environments["environments<br/>7 files, 749 lines"]
    evolution["evolution<br/>10 files, 2380 lines"]
    grounding["grounding<br/>9 files, 1333 lines"]
    lattice["lattice<br/>5 files, 704 lines"]
    maintenance["maintenance<br/>2 files, 295 lines"]
    persistence["persistence<br/>2 files, 618 lines"]
    plasticity["plasticity<br/>5 files, 428 lines"]
    predictive["predictive<br/>2 files, 186 lines"]
    sensors["sensors<br/>1 files, 159 lines"]
    services["services<br/>2 files, 31 lines"]
    sim["sim<br/>7 files, 671 lines"]
    simulation["simulation<br/>3 files, 401 lines"]
    soma["soma<br/>4 files, 1399 lines"]
    sovereign["sovereign<br/>5 files, 571 lines"]
    startup["startup<br/>4 files, 546 lines"]
    unknowns["unknowns<br/>4 files, 325 lines"]
    architecture_quality["architecture_quality<br/>3 files, 671 lines"]
    audit["audit<br/>7 files, 649 lines"]
    body["body<br/>22 files, 1379 lines"]
    consent["consent<br/>2 files, 167 lines"]
    context["context<br/>5 files, 1438 lines"]
    creativity["creativity<br/>2 files, 801 lines"]
    curriculum["curriculum<br/>7 files, 657 lines"]
    cybernetics["cybernetics<br/>6 files, 1134 lines"]
    evals["evals<br/>2 files, 293 lines"]
    factory["factory<br/>8 files, 758 lines"]
    guardians["guardians<br/>7 files, 945 lines"]
    initializers["initializers<br/>3 files, 199 lines"]
    intent["intent<br/>2 files, 688 lines"]
    middleware["middleware<br/>2 files, 254 lines"]
    networking["networking<br/>3 files, 920 lines"]
    quantum["quantum<br/>3 files, 527 lines"]
    research_core["research_core<br/>5 files, 580 lines"]
    safety["safety<br/>3 files, 629 lines"]
    session["session<br/>3 files, 642 lines"]
    skill_management["skill_management<br/>1 files, 367 lines"]
    sovereignty["sovereignty<br/>4 files, 2095 lines"]
    systems["systems<br/>3 files, 256 lines"]
    tools["tools<br/>10 files, 1122 lines"]
    transparency["transparency<br/>2 files, 317 lines"]
    twins["twins<br/>1 files, 97 lines"]
    welfare["welfare<br/>7 files, 228 lines"]
    worlds["worlds<br/>5 files, 1930 lines"]
    actuation["actuation<br/>9 files, 350 lines"]
    audits["audits<br/>2 files, 267 lines"]
    control["control<br/>3 files, 640 lines"]
    core_root["core_root<br/>49 files, 24518 lines"]
    council["council<br/>5 files, 533 lines"]
    forge["forge<br/>8 files, 325 lines"]
    lab["lab<br/>7 files, 482 lines"]
    latent["latent<br/>1 files, 56 lines"]
    mission["mission<br/>4 files, 472 lines"]
    multimodal["multimodal<br/>2 files, 185 lines"]
    neuroweb["neuroweb<br/>5 files, 367 lines"]
    ontology["ontology<br/>2 files, 169 lines"]
    pipeline["pipeline<br/>3 files, 217 lines"]
    play["play<br/>1 files, 228 lines"]
    providers["providers<br/>6 files, 1314 lines"]
    reproducibility["reproducibility<br/>2 files, 497 lines"]
    science["science<br/>1 files, 139 lines"]
    swarm["swarm<br/>5 files, 396 lines"]
    temporal["temporal<br/>3 files, 1507 lines"]
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
    runtime --> learning
    runtime --> memory
    runtime --> observability
    runtime --> organism
    runtime --> perception
    runtime --> persistence
    runtime --> phases
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
    runtime --> workspace
    utils --> consciousness
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
    brain --> constitution
    brain --> continuity
    brain --> conversation
    brain --> discovery
    brain --> epistemics
    brain --> health
    brain --> identity
    brain --> introspection
    brain --> kernel
    brain --> knowledge
    brain --> learning
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
    brain --> self_modification
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
    memory --> knowledge
    memory --> observability
    memory --> phases
    memory --> resilience
    memory --> runtime
    memory --> security
    memory --> social
    memory --> utils
    memory --> values
    consciousness --> actuators
    consciousness --> adaptation
    consciousness --> affect
    consciousness --> agency
    consciousness --> being
    consciousness --> brain
    consciousness --> constitution
    consciousness --> continuity
    consciousness --> coordinators
    consciousness --> evaluation
    consciousness --> ghost
    consciousness --> goals
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
    consciousness --> world
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
    health --> brain
    health --> memory
    health --> runtime
    health --> state
    health --> utils
    agency --> adaptation
    agency --> affect
    agency --> agi
    agency --> autonomy
    agency --> brain
    agency --> cognition
    agency --> consciousness
    agency --> constitution
    agency --> continuity
    agency --> conversation
    agency --> governance
    agency --> health
    agency --> identity
    agency --> knowledge
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
    observability --> health
    observability --> memory
    observability --> runtime
    governance --> actuators
    governance --> being
    governance --> brain
    governance --> consciousness
    governance --> identity
    governance --> memory
    governance --> observability
    governance --> resilience
    governance --> runtime
    governance --> tools
    governance --> utils
    adaptation --> actuators
    adaptation --> affect
    adaptation --> being
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
    security --> affect
    security --> agency
    security --> brain
    security --> consciousness
    security --> identity
    security --> memory
    security --> perception
    security --> runtime
    security --> utils
    senses --> affect
    senses --> brain
    senses --> consciousness
    senses --> constitution
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
    identity --> agency
    identity --> brain
    identity --> governance
    identity --> organism
    identity --> runtime
    identity --> utils
    self_modification --> architecture_quality
    self_modification --> bus
    self_modification --> ethics
    self_modification --> governance
    self_modification --> memory
    self_modification --> ops
    self_modification --> resilience
    self_modification --> runtime
    self_modification --> skills
    self_modification --> utils
    conversation --> autonomy
    conversation --> brain
    conversation --> consciousness
    conversation --> constitution
    conversation --> health
    conversation --> introspection
    conversation --> memory
    conversation --> organism
    conversation --> runtime
    conversation --> social
    conversation --> utils
    cognition --> actuators
    cognition --> affect
    cognition --> agency
    cognition --> brain
    cognition --> consciousness
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
    state --> bus
    state --> constitution
    state --> governance
    state --> memory
    state --> motivation
    state --> runtime
    state --> unity
    state --> utils
    state --> values
    perception --> brain
    perception --> capabilities
    perception --> media
    perception --> phenomenal_substrate
    perception --> resilience
    perception --> runtime
    perception --> security
    perception --> senses
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
    orchestrator --> cognition
    orchestrator --> cognitive
    orchestrator --> collective
    orchestrator --> consciousness
    orchestrator --> constitution
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
    orchestrator --> voice
    orchestrator --> world_model
    skills --> actuators
    skills --> advanced_cognition
    skills --> affect
    skills --> being
    skills --> brain
    skills --> capabilities
    skills --> consciousness
    skills --> consent
    skills --> conversation
    skills --> embodiment
    skills --> executive
    skills --> governance
    skills --> knowledge
    skills --> learning
    skills --> memory
    skills --> perception
    skills --> quantum
    skills --> runtime
    skills --> sandbox
    skills --> search
    skills --> security
    skills --> self_improvement
    skills --> self_modification
    skills --> senses
    skills --> sovereign
    skills --> utils
    skills --> worlds
    world_model --> advanced_cognition
    world_model --> brain
    world_model --> cognition
    world_model --> constitution
    world_model --> health
    world_model --> resilience
    world_model --> runtime
    world_model --> values
    autonomy --> affect
    autonomy --> agency
    autonomy --> brain
    autonomy --> consciousness
    autonomy --> constitution
    autonomy --> continuity
    autonomy --> conversation
    autonomy --> conversational
    autonomy --> discovery
    autonomy --> executive
    autonomy --> governance
    autonomy --> health
    autonomy --> memory
    autonomy --> observability
    autonomy --> planning
    autonomy --> resource
    autonomy --> runtime
    autonomy --> skills
    autonomy --> sleep
    autonomy --> state
    autonomy --> utils
    autonomy --> voice
    autonomy --> world_model
    epistemics --> being
    epistemics --> brain
    epistemics --> observability
    epistemics --> reasoning
    epistemics --> runtime
    epistemics --> skills
    epistemics --> utils
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
    continuity --> identity
    continuity --> organism
    continuity --> runtime
    organism --> adaptation
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
    values --> agency
    values --> governance
    values --> runtime
    values --> social
    values --> utils
    being --> agency
    being --> consciousness
    being --> epistemics
    being --> governance
    being --> observability
    being --> runtime
    learning --> brain
    learning --> consciousness
    learning --> ghost
    learning --> introspection
    learning --> memory
    learning --> orchestrator
    learning --> promotion
    learning --> reasoning
    learning --> runtime
    learning --> self_modification
    learning --> skills
    learning --> tasks
    learning --> utils
    learning --> world_model
    knowledge --> brain
    knowledge --> runtime
    knowledge --> utils
    morality --> autonomy
    morality --> brain
    morality --> consciousness
    morality --> perception
    morality --> runtime
    morality --> utils
    tasks --> runtime
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
    capabilities --> adapters
    capabilities --> brain
    capabilities --> constitution
    capabilities --> governance
    capabilities --> knowledge
    capabilities --> memory
    capabilities --> perception
    capabilities --> phenomenal_substrate
    capabilities --> planning
    capabilities --> runtime
    capabilities --> security
    capabilities --> self
    capabilities --> skills
    capabilities --> utils
    capabilities --> voice
    introspection --> cognition
    introspection --> health
    introspection --> resilience
    introspection --> runtime
    introspection --> security
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
    phases --> self_modification
    phases --> skills
    phases --> social
    phases --> somatic
    phases --> state
    phases --> unity
    phases --> utils
    phases --> voice
    planning --> brain
    planning --> capabilities
    planning --> collective
    planning --> data
    planning --> runtime
    planning --> utils
    reasoning --> observability
    reasoning --> planning
    reasoning --> runtime
    reasoning --> utils
    self --> affect
    self --> bus
    self --> consciousness
    self --> epistemics
    self --> memory
    self --> runtime
    self --> security
    self --> senses
    self --> state
    self --> utils
    voice --> brain
    voice --> conversational
    voice --> executive
    voice --> managers
    voice --> resilience
    voice --> runtime
    voice --> senses
    voice --> utils
    world --> governance
    world --> runtime
    bus --> capabilities
    bus --> resilience
    bus --> runtime
    bus --> utils
    discovery --> cognition
    discovery --> memory
    discovery --> observability
    discovery --> runtime
    discovery --> self_modification
    discovery --> unknowns
    kernel --> agency
    kernel --> brain
    kernel --> cognition
    kernel --> consciousness
    kernel --> continuity
    kernel --> cybernetics
    kernel --> executive
    kernel --> health
    kernel --> introspection
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
    autonomic --> embodiment
    autonomic --> orchestrator
    autonomic --> runtime
    autonomic --> utils
    cognitive --> brain
    cognitive --> health
    cognitive --> phases
    cognitive --> runtime
    cognitive --> utils
    coordinators --> autonomy
    coordinators --> brain
    coordinators --> continuity
    coordinators --> conversation
    coordinators --> environment
    coordinators --> epistemics
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
    coordinators --> sleep
    coordinators --> somatic
    coordinators --> tasks
    coordinators --> utils
    coordinators --> world_model
    embodiment --> agency
    embodiment --> consciousness
    embodiment --> environments
    embodiment --> ethics
    embodiment --> governance
    embodiment --> organism
    embodiment --> runtime
    embodiment --> utils
    ethics --> brain
    ethics --> morality
    ethics --> runtime
    ethics --> utils
    managers --> autonomic
    managers --> brain
    managers --> bus
    managers --> cognition
    managers --> collective
    managers --> constitution
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
    meta --> adaptation
    meta --> runtime
    meta --> utils
    resource --> observability
    resource --> resilience
    resource --> runtime
    sleep --> adaptation
    sleep --> affect
    sleep --> brain
    sleep --> identity
    sleep --> memory
    sleep --> runtime
    sleep --> systems
    sleep --> world_model
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
    agi --> adaptation
    agi --> brain
    agi --> constitution
    agi --> conversation
    agi --> embodiment
    agi --> epistemics
    agi --> executive
    agi --> grounding
    agi --> health
    agi --> runtime
    agi --> utils
    agi --> world_model
    collective --> adaptation
    collective --> agency
    collective --> brain
    collective --> planning
    collective --> runtime
    collective --> utils
    data --> runtime
    evaluation --> brain
    evaluation --> conversation
    evaluation --> learning
    evaluation --> promotion
    evaluation --> runtime
    goals --> agency
    goals --> autonomy
    goals --> brain
    goals --> runtime
    goals --> state
    goals --> utils
    goals --> values
    motivation --> brain
    motivation --> consciousness
    motivation --> constitution
    motivation --> health
    motivation --> runtime
    motivation --> utils
    motivation --> values
    phenomenal_substrate --> runtime
    promotion --> runtime
    sandbox --> runtime
    self_improvement --> brain
    self_improvement --> discovery
    self_improvement --> llm
    self_improvement --> runtime
    self_improvement --> self_modification
    self_improvement --> skills
    somatic --> media
    somatic --> memory
    somatic --> observability
    somatic --> runtime
    somatic --> utils
    somatic --> world_model
    adapters --> agency
    adapters --> brain
    adapters --> runtime
    advanced_cognition --> reasoning
    advanced_cognition --> runtime
    conversational --> memory
    conversational --> runtime
    conversational --> social
    db --> runtime
    environment --> advanced_cognition
    environment --> brain
    environment --> consciousness
    environment --> environments
    environment --> executive
    environment --> memory
    environment --> perception
    environment --> runtime
    ghost --> memory
    ghost --> runtime
    ghost --> self
    llm --> brain
    morphogenesis --> adaptation
    morphogenesis --> memory
    morphogenesis --> resilience
    morphogenesis --> runtime
    morphogenesis --> self_modification
    pneuma --> affect
    pneuma --> runtime
    pneuma --> utils
    search --> capabilities
    search --> knowledge
    search --> runtime
    search --> utils
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
    environments --> environment
    environments --> perception
    environments --> runtime
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
    sim --> twins
    sim --> utils
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
    audit --> epistemics
    audit --> runtime
    body --> capabilities
    body --> perception
    body --> runtime
    body --> security
    context --> conversation
    context --> runtime
    creativity --> memory
    creativity --> runtime
    curriculum --> runtime
    cybernetics --> cognitive
    cybernetics --> kernel
    cybernetics --> runtime
    cybernetics --> utils
    evals --> runtime
    factory --> runtime
    guardians --> brain
    guardians --> morality
    guardians --> runtime
    guardians --> tasks
    guardians --> utils
    initializers --> adaptation
    initializers --> consciousness
    initializers --> introspection
    initializers --> memory
    initializers --> meta
    initializers --> perception
    initializers --> runtime
    initializers --> senses
    initializers --> utils
    intent --> brain
    intent --> epistemics
    intent --> runtime
    intent --> utils
    middleware --> runtime
    networking --> agency
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
    tools --> resilience
    tools --> runtime
    tools --> sandbox
    tools --> skills
    transparency --> runtime
    worlds --> runtime
    actuation --> runtime
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
    core_root --> cognition
    core_root --> coherence
    core_root --> consciousness
    core_root --> constitution
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
    core_root --> llm
    core_root --> media
    core_root --> memory
    core_root --> meta
    core_root --> motivation
    core_root --> observability
    core_root --> orchestrator
    core_root --> organism
    core_root --> phases
    core_root --> planning
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
    pipeline --> runtime
    play --> consciousness
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
    providers --> resilience
    providers --> runtime
    providers --> self_modification
    providers --> senses
    providers --> services
    providers --> sleep
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
    temporal --> runtime
    temporal --> utils
```

## Core Subsystem Stats

| Subsystem | Files | Lines | Bytes | Deps Out | Deps In |
| --- | ---: | ---: | ---: | ---: | ---: |
| consciousness | 149 | 68953 | 2906214 | 41 | 36 |
| brain | 162 | 64203 | 2768474 | 49 | 57 |
| runtime | 146 | 47478 | 1753552 | 49 | 133 |
| skills | 94 | 28365 | 1169822 | 35 | 12 |
| core_root | 49 | 24518 | 1050848 | 65 | 0 |
| memory | 94 | 23556 | 952734 | 22 | 41 |
| orchestrator | 45 | 21144 | 932232 | 103 | 12 |
| phases | 29 | 20096 | 909585 | 35 | 7 |
| agency | 49 | 18870 | 768607 | 33 | 23 |
| resilience | 66 | 16902 | 692057 | 20 | 32 |
| learning | 41 | 13244 | 535529 | 20 | 9 |
| adaptation | 28 | 13232 | 530131 | 23 | 17 |
| self_modification | 35 | 12733 | 504168 | 15 | 16 |
| autonomy | 29 | 11861 | 487872 | 30 | 11 |
| conversation | 19 | 11441 | 426613 | 19 | 15 |
| capabilities | 20 | 11426 | 455271 | 18 | 7 |
| perception | 29 | 11205 | 452964 | 14 | 13 |
| cognitive | 12 | 9234 | 374849 | 10 | 5 |
| environment | 83 | 9202 | 360303 | 11 | 3 |
| security | 39 | 9009 | 358498 | 14 | 17 |
| cognition | 24 | 8641 | 363821 | 20 | 14 |
| social | 21 | 7779 | 323084 | 16 | 11 |
| senses | 27 | 7004 | 291998 | 21 | 17 |
| being | 28 | 6931 | 274980 | 9 | 9 |
| utils | 45 | 6745 | 261910 | 18 | 70 |
| kernel | 11 | 6530 | 272718 | 25 | 6 |
| architect | 25 | 5743 | 240043 | 10 | 2 |
| self_improvement | 14 | 5586 | 232691 | 7 | 4 |
| ops | 18 | 5377 | 211471 | 17 | 6 |
| reasoning | 12 | 5307 | 212352 | 7 | 7 |
| epistemics | 14 | 4834 | 197700 | 10 | 11 |
| coordinators | 10 | 4604 | 213732 | 33 | 5 |
| state | 9 | 4440 | 185757 | 12 | 14 |
| voice | 10 | 4427 | 193665 | 12 | 7 |
| affect | 12 | 4161 | 186798 | 14 | 17 |
| planning | 9 | 4035 | 162980 | 9 | 7 |
| governance | 10 | 3593 | 148520 | 16 | 19 |
| goals | 10 | 3535 | 151593 | 11 | 4 |
| world_model | 11 | 3290 | 138643 | 12 | 12 |
| executive | 11 | 3278 | 135213 | 17 | 11 |
| embodiment | 15 | 3139 | 122211 | 10 | 5 |
| evaluation | 16 | 2983 | 106661 | 6 | 4 |
| advanced_cognition | 13 | 2905 | 118305 | 3 | 3 |
| somatic | 6 | 2894 | 113333 | 12 | 4 |
| morphogenesis | 12 | 2875 | 111602 | 7 | 3 |
| unity | 11 | 2791 | 117739 | 8 | 5 |
| identity | 18 | 2734 | 113755 | 9 | 16 |
| observability | 11 | 2583 | 97491 | 9 | 20 |
| bus | 6 | 2545 | 105116 | 6 | 6 |
| conversational | 4 | 2434 | 101480 | 5 | 3 |
| evolution | 10 | 2380 | 95065 | 9 | 2 |
| actuators | 10 | 2333 | 91577 | 14 | 7 |
| self | 7 | 2196 | 91293 | 12 | 7 |
| introspection | 8 | 2164 | 86447 | 9 | 7 |
| discovery | 7 | 2134 | 90979 | 8 | 6 |
| sovereignty | 4 | 2095 | 89078 | 11 | 1 |
| collective | 6 | 2079 | 85501 | 7 | 4 |
| ghost | 6 | 1961 | 79701 | 6 | 3 |
| values | 15 | 1953 | 80675 | 8 | 10 |
| organism | 9 | 1949 | 73328 | 17 | 10 |
| worlds | 5 | 1930 | 80501 | 3 | 1 |
| search | 2 | 1758 | 66983 | 8 | 3 |
| agi | 6 | 1562 | 65083 | 13 | 4 |
| temporal | 3 | 1507 | 50941 | 2 | 0 |
| world | 24 | 1483 | 54104 | 3 | 7 |
| context | 5 | 1438 | 55868 | 2 | 1 |
| soma | 4 | 1399 | 54836 | 6 | 2 |
| body | 22 | 1379 | 49082 | 6 | 1 |
| grounding | 9 | 1333 | 49886 | 6 | 2 |
| morality | 16 | 1327 | 51614 | 9 | 8 |
| providers | 6 | 1314 | 58732 | 42 | 0 |
| meta | 7 | 1276 | 48151 | 5 | 5 |
| pneuma | 7 | 1257 | 47662 | 3 | 3 |
| workspace | 9 | 1242 | 45306 | 3 | 3 |
| autonomic | 5 | 1237 | 49903 | 8 | 5 |
| motivation | 7 | 1209 | 51041 | 11 | 4 |
| cybernetics | 6 | 1134 | 45291 | 6 | 1 |
| tools | 10 | 1122 | 42265 | 6 | 1 |
| health | 5 | 1121 | 43895 | 7 | 26 |
| knowledge | 10 | 1112 | 40321 | 6 | 8 |
| phenomenal_substrate | 11 | 1042 | 41881 | 1 | 4 |
| adapters | 5 | 1018 | 38923 | 6 | 3 |
| supervisor | 3 | 976 | 37664 | 3 | 5 |
| managers | 6 | 959 | 40802 | 25 | 5 |
| guardians | 7 | 945 | 39788 | 8 | 1 |
| promotion | 6 | 936 | 31616 | 1 | 4 |
| networking | 3 | 920 | 34350 | 4 | 1 |
| sleep | 9 | 858 | 37487 | 10 | 5 |
| creativity | 2 | 801 | 33361 | 3 | 1 |
| factory | 8 | 758 | 29090 | 3 | 1 |
| environments | 7 | 749 | 31176 | 3 | 2 |
| lattice | 5 | 704 | 26089 | 0 | 2 |
| intent | 2 | 688 | 27268 | 5 | 1 |
| architecture_quality | 3 | 671 | 24422 | 0 | 1 |
| sim | 7 | 671 | 25037 | 6 | 2 |
| curriculum | 7 | 657 | 21995 | 1 | 1 |
| data | 3 | 651 | 22377 | 2 | 4 |
| audit | 7 | 649 | 26024 | 4 | 1 |
| session | 3 | 642 | 25682 | 2 | 1 |
| control | 3 | 640 | 23151 | 4 | 0 |
| safety | 3 | 629 | 25776 | 3 | 1 |
| resource | 4 | 624 | 23470 | 4 | 5 |
| persistence | 2 | 618 | 25034 | 3 | 2 |
| sandbox | 4 | 610 | 21744 | 1 | 4 |
| tasks | 5 | 597 | 20475 | 4 | 8 |
| db | 4 | 584 | 22537 | 2 | 3 |
| ethics | 2 | 580 | 23482 | 6 | 5 |
| research_core | 5 | 580 | 22543 | 8 | 1 |
| sovereign | 5 | 571 | 20110 | 3 | 2 |
| startup | 4 | 546 | 19778 | 12 | 2 |
| council | 5 | 533 | 21566 | 4 | 0 |
| quantum | 3 | 527 | 20162 | 0 | 1 |
| reproducibility | 2 | 497 | 18141 | 1 | 0 |
| lab | 7 | 482 | 19394 | 3 | 0 |
| mission | 4 | 472 | 17806 | 1 | 0 |
| plasticity | 5 | 428 | 15395 | 2 | 2 |
| coherence | 2 | 407 | 19530 | 6 | 2 |
| simulation | 3 | 401 | 15916 | 7 | 2 |
| swarm | 5 | 396 | 15170 | 6 | 0 |
| neuroweb | 5 | 367 | 14205 | 5 | 0 |
| skill_management | 1 | 367 | 17972 | 5 | 1 |
| actuation | 9 | 350 | 11972 | 2 | 0 |
| verification | 4 | 350 | 13177 | 2 | 3 |
| media | 2 | 336 | 11674 | 0 | 4 |
| forge | 8 | 325 | 11877 | 2 | 0 |
| unknowns | 4 | 325 | 11829 | 3 | 2 |
| transparency | 2 | 317 | 12256 | 1 | 1 |
| maintenance | 2 | 295 | 10758 | 3 | 2 |
| evals | 2 | 293 | 10146 | 2 | 1 |
| audits | 2 | 267 | 9492 | 3 | 0 |
| systems | 3 | 256 | 9861 | 3 | 1 |
| middleware | 2 | 254 | 11026 | 2 | 1 |
| continuity | 7 | 238 | 8314 | 4 | 10 |
| play | 1 | 228 | 8774 | 4 | 0 |
| welfare | 7 | 228 | 8034 | 0 | 1 |
| pipeline | 3 | 217 | 6684 | 1 | 0 |
| llm | 3 | 200 | 7254 | 1 | 3 |
| initializers | 3 | 199 | 9095 | 10 | 1 |
| predictive | 2 | 186 | 7113 | 4 | 2 |
| multimodal | 2 | 185 | 6591 | 1 | 0 |
| ontology | 2 | 169 | 5381 | 0 | 0 |
| consent | 2 | 167 | 5514 | 0 | 1 |
| sensors | 1 | 159 | 5963 | 2 | 2 |
| science | 1 | 139 | 5947 | 4 | 0 |
| twins | 1 | 97 | 3626 | 0 | 1 |
| latent | 1 | 56 | 2337 | 0 | 0 |
| services | 2 | 31 | 1171 | 1 | 2 |
| constitution | 1 | 25 | 795 | 0 | 17 |

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

- Unique services retrieved: 394
- Unique services registered: 275
- Services retrieved without detected registration: 233

### Top Fetched Services

| Service | Gets | Registrations |
| --- | ---: | ---: |
| orchestrator | 61 | 3 |
| inference_gate | 43 | 4 |
| cognitive_engine | 37 | 2 |
| llm_router | 37 | 2 |
| affect_engine | 35 | 1 |
| capability_engine | 33 | 2 |
| memory_facade | 28 | 1 |
| liquid_substrate | 24 | 1 |
| conscious_substrate | 24 | 1 |
| free_energy_engine | 22 | 0 |
| drive_engine | 21 | 1 |
| world_state | 21 | 0 |
| mycelial_network | 20 | 1 |
| global_workspace | 20 | 1 |
| homeostasis | 19 | 1 |
| state_repository | 18 | 1 |
| goal_engine | 18 | 0 |
| knowledge_graph | 17 | 0 |
| qualia_synthesizer | 17 | 2 |
| episodic_memory | 16 | 1 |

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
- `being_runtime` fetched 2 time(s)
- `belief_challenger` fetched 2 time(s)
- `belief_engine` fetched 1 time(s)
- `belief_system` fetched 1 time(s)
- `bicameral_advisory` fetched 1 time(s)
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
- `cloud_body` fetched 1 time(s)
- `code_repair` fetched 1 time(s)
- `cognitive_kernel` fetched 2 time(s)
- `coherence_report` fetched 1 time(s)
- `cold_store` fetched 1 time(s)
- `concept_bridge` fetched 2 time(s)
- `concept_linker` fetched 1 time(s)
- `config` fetched 1 time(s)
- `consciousness_bridge` fetched 2 time(s)
- `consciousness_evidence` fetched 1 time(s)
- `constitution` fetched 1 time(s)

## Operational Authority Map

| Surface | Calls | Files | Owner Calls | Review Candidates |
| --- | ---: | ---: | ---: | ---: |
| UnifiedWill decisions | 62 | 33 | 2 | 60 |
| Memory writes | 340 | 136 | 52 | 288 |
| State mutation | 469 | 174 | 9 | 460 |
| Tool execution | 102 | 54 | 6 | 96 |
| Self-modification and patching | 14 | 12 | 1 | 13 |
| LLM inference | 277 | 168 | 74 | 203 |
| External I/O | 121 | 44 | 8 | 113 |

### UnifiedWill decisions

Calls that can ask the single will authority to approve action.

Review candidates:
- `core/actuators/actuator_synthesis.py:183` [actuators] `get_will` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:183` [actuators] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/adaptive_immunity.py:2103` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/adaptive_immunity.py:2103` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:624` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:624` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:204` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:204` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/agency/agency_bus.py:83` [agency] `get_will` - _auto_decision = get_will().decide(
- `core/agency/agency_bus.py:83` [agency] `get_will.decide` - _auto_decision = get_will().decide(
- `core/agency/hierarchical_agency.py:402` [agency] `get_will` - decision = get_will().decide(
- `core/agency/hierarchical_agency.py:402` [agency] `get_will.decide` - decision = get_will().decide(
- `core/autonomy/genuine_refusal.py:303` [autonomy] `will.decide` - decision = will.decide(content, source="genuine_refusal", domain=domain, priority=0.8, context=ctx)
- `core/autonomy/self_modification.py:315` [autonomy] `will.decide` - decision = will.decide(
- `core/cognitive/autopoiesis.py:961` [cognitive] `will.decide` - decision = will.decide(
- `core/consciousness/parallel_branches.py:736` [consciousness] `will.decide` - decision = will.decide(
- `core/environment/governance_bridge.py:47` [environment] `self.will_gateway.decide` - will_decision = await self.will_gateway.decide(intent)
- `core/goals/goal_engine.py:1046` [goals] `will.decide` - decision = will.decide(
- `core/governance/will_gate.py:109` [governance] `will.decide` - decision = will.decide(
- `core/governance/will_gate.py:160` [governance] `will.decide` - decision = will.decide(
- `core/initiative_synthesis.py:831` [core_root] `get_will` - decision = get_will().decide(
- `core/initiative_synthesis.py:831` [core_root] `get_will.decide` - decision = get_will().decide(
- `core/learning/compounding_scheduler.py:328` [learning] `get_will` - decision = get_will().decide(
- `core/learning/compounding_scheduler.py:328` [learning] `get_will.decide` - decision = get_will().decide(
- `core/learning/genuine_learning_pipeline.py:657` [learning] `get_will` - decision = get_will().decide(

### Memory writes

Calls that can create durable or semantically promoted memory.

Review candidates:
- `core/actuators/doc_ingest.py:81` [actuators] `get_memory_write_gateway` - gateway = get_memory_write_gateway()
- `core/actuators/doc_ingest.py:95` [actuators] `MemoryWriteRequest` - MemoryWriteRequest(
- `core/actuators/doc_ingest.py:106` [actuators] `memory_facade.add_memory` - maybe_result = memory_facade.add_memory(text=text, metadata=metadata)
- `core/adaptation/abstraction_engine.py:125` [adaptation] `MemoryWriteReceipt` - MemoryWriteReceipt(
- `core/adaptation/abstraction_engine.py:144` [adaptation] `memory_facade.store` - await memory_facade.store(
- `core/adaptation/adaptive_immunity.py:1266` [adaptation] `self._cells.append` - self._cells.append(memory)
- `core/advanced_cognition/continual_learning_stability.py:94` [advanced_cognition] `self._persist_memory` - self._persist_memory(rec)
- `core/advanced_cognition/continual_learning_stability.py:98` [advanced_cognition] `self.store_memory` - return self.store_memory(
- `core/advanced_cognition/continual_learning_stability.py:208` [advanced_cognition] `scored.append` - scored.append((score, memory))
- `core/advanced_cognition/continual_learning_stability.py:313` [advanced_cognition] `self._append_jsonl` - self._append_jsonl(self.state_dir / "memory.jsonl", rec.to_dict())
- `core/affect/phenomenal_integration.py:596` [affect] `memory.set_write_weights` - memory.set_write_weights(state.memory_weights)
- `core/agency/autonomous_task_engine.py:1007` [agency] `self._mycelial.add_edge` - await self._mycelial.add_edge(context["source_memory"], goal[:40])
- `core/agency/latent_distiller.py:60` [agency] `self.memory.store_memory` - await self.memory.store_memory(
- `core/architect/code_graph.py:679` [architect] `effects.add` - effects.add("memory_write")
- `core/architect/safe_boot_harness.py:79` [architect] `probe_memory_write_read` - memory = await probe_memory_write_read(tmp_root=root / "memory")
- `core/architect/smell_detector.py:177` [architect] `self._effect_smell` - smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, F
- `core/architect/smell_detector.py:177` [architect] `smells.append` - smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, F
- `core/autonomy/autonomous_initiative_loop.py:1220` [autonomy] `memory.store` - await memory.store(
- `core/autonomy/autonomous_initiative_loop.py:1230` [autonomy] `logger.debug` - logger.debug("Social observation memory write failed: %s", exc)
- `core/autonomy/autonomous_research_orchestrator.py:155` [autonomy] `MemoryPersister` - self._persister = persister or MemoryPersister()
- `core/autonomy/initiative_overflow.py:156` [autonomy] `logger.debug` - logger.debug("Skill gap memory write failed: %s", exc)
- `core/autonomy/initiative_overflow.py:166` [autonomy] `memory.store_sync` - memory.store_sync(
- `core/autonomy/personhood_engine.py:199` [autonomy] `state.cognition.working_memory.append` - state.cognition.working_memory.append(
- `core/autonomy/research_cycle.py:768` [autonomy] `state.cognition.long_term_memory.append` - state.cognition.long_term_memory.append(
- `core/autonomy/research_cycle.py:786` [autonomy] `hasattr` - if memory_facade is not None and hasattr(memory_facade, "add_memory"):

### State mutation

Calls that can mutate runtime, identity, repository, or persistent state.

Review candidates:
- `core/adaptation/adaptive_immunity.py:954` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:1302` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:1486` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:1593` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:2347` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:2665` [adaptation] `atomic_write_text` - atomic_write_text(self._state_path, json.dumps(payload, indent=2), encoding="utf-8")
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
- `core/agency/agency_core.py:196` [agency] `get_registry.update` - await get_registry().update(active_shards=len(self.active_shards))
- `core/agency/agency_core.py:203` [agency] `setattr` - on_unscheduled=lambda: setattr(self, "_registry_shards_update_pending", False),
- `core/agency/agency_core.py:944` [agency] `virtual_body.__dict__.update` - virtual_body.__dict__.update(snapshot)
- `core/agency/agency_core.py:1138` [agency] `get_registry.update` - await get_registry().update(
- `core/agency/agency_core.py:1147` [agency] `_run_registry_update` - _run_registry_update(),
- `core/agency/agency_core.py:1149` [agency] `setattr` - on_unscheduled=lambda: setattr(self, "_registry_update_pending", False),

### Tool execution

Calls that can execute tools, skills, shells, browsers, or external actions.

Review candidates:
- `core/actuators/actuator_registry.py:375` [actuators] `self.operator.execute_synthesized_tool` - res = self.operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/code_execution_actuator.py:98` [actuators] `operator.execute_synthesized_tool` - res = operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/web_actuators.py:125` [actuators] `skill.execute` - return await skill.execute({"mode": "browse", "url": url}, {})
- `core/agency/agency_core.py:537` [agency] `self._execute_shard_tool` - tasks.append(self._execute_shard_tool(name, payload))
- `core/agency/agency_orchestrator.py:369` [agency] `execute` - await execute(proposal, state_snapshot, receipt.capability_token or "")
- `core/agency/autonomous_task_engine.py:515` [agency] `orchestrator.execute_tool` - return await orchestrator.execute_tool(tool_name, args, **kwargs)
- `core/agency/autonomous_task_engine.py:2841` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:2844` [agency] `orch.execute_tool` - return await orch.execute_tool("web_search", {"query": query})
- `core/agency/autonomous_task_engine.py:2859` [agency] `orch.execute_tool` - result = await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:2863` [agency] `orch.execute_tool` - result = await orch.execute_tool("run_python", {"code": code})
- `core/agency/desktop_planner.py:56` [agency] `skill.execute` - await skill.execute({"action": action, **params}, {})
- `core/agency/skill_library.py:199` [agency] `tool_orchestrator.execute_tool` - result = await tool_orchestrator.execute_tool(step.tool_name, resolved_args)
- `core/agi/curiosity_explorer.py:274` [agi] `orchestrator.execute_tool` - orchestrator.execute_tool(
- `core/autonomy/autonomous_initiative_loop.py:674` [autonomy] `capability_engine.execute` - scan_result = await capability_engine.execute(
- `core/autonomy/autonomous_initiative_loop.py:720` [autonomy] `capability_engine.execute` - test_result = await capability_engine.execute(
- `core/autonomy/autonomous_initiative_loop.py:759` [autonomy] `capability_engine.execute` - proposal_result = await capability_engine.execute(
- `core/autonomy/autonomous_initiative_loop.py:1197` [autonomy] `skill.execute` - return await skill.execute(EmailInput(**payload), {})
- `core/autonomy/autonomous_initiative_loop.py:1209` [autonomy] `skill.execute` - return await skill.execute(RedditInput(**payload), {})
- `core/autonomy/behavior_controller.py:99` [autonomy] `self.orchestrator.execute_tool` - self.orchestrator.execute_tool(tool_name, arguments), target_loop
- `core/autonomy/behavior_controller.py:104` [autonomy] `self.orchestrator.execute_tool` - self.orchestrator.execute_tool(tool_name, arguments)
- `core/autonomy/proactive_presence.py:609` [autonomy] `tool_orch.execute_tool` - result = await tool_orch.execute_tool("web_search", {"query": "latest world news summary"})
- `core/autonomy/research_cycle.py:613` [autonomy] `self.orchestrator.execute_tool` - lambda name=tool_name, **kw: self.orchestrator.execute_tool(name, kw, origin="research_cycle")
- `core/autonomy/research_cycle.py:620` [autonomy] `self.orchestrator.execute_tool` - lambda name=tool_name, **kw: self.orchestrator.execute_tool(name, kw, origin="research_cycle")
- `core/autonomy/research_cycle.py:917` [autonomy] `self.orchestrator.execute_tool` - result = await self.orchestrator.execute_tool(
- `core/brain/llm/function_calling_adapter.py:131` [brain] `skill.execute` - result = await skill.execute(args, {"source": "autonomous_brain"})

### Self-modification and patching

Calls that can generate, validate, apply, or promote code changes.

Review candidates:
- `core/architect/governor.py:140` [architect] `self.promotion_governor.promote` - decision = self.promotion_governor.promote(plan, shadow, proof, rollback)
- `core/evolution/optimizer.py:56` [evolution] `patch.apply` - success = await patch.apply(signature)
- `core/evolution/optimizer.py:67` [evolution] `cog_patch.apply` - if await cog_patch.apply(signature):
- `core/factory/software_factory.py:115` [factory] `self.writer.write_patch` - patch = await self.writer.write_patch(change, repo_path)
- `core/guardians/airlock.py:81` [guardians] `async_atomic_write_text` - await async_atomic_write_text(patch_file, diff_patch, encoding="utf-8")
- `core/kernel/upgrades_10x.py:345` [kernel] `self._safe_self_modify` - await self._safe_self_modify(state)
- `core/orchestrator/mixins/boot/boot_autonomy.py:971` [orchestrator] `apply_presence_patch` - apply_presence_patch(self)
- `core/runtime/safe_mode.py:140` [runtime] `apply_orchestrator_patches` - apply_orchestrator_patches(orchestrator, safe_mode=bool(enabled))
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
- `core/adaptation/epistemic_humility.py:162` [adaptation] `llm.chat` - response = await llm.chat(
- `core/adaptation/heuristic_synthesizer.py:129` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/star_reasoner.py:365` [adaptation] `llm.think` - result = await asyncio.wait_for(llm.think(prompt), timeout=self.RATIONALIZATION_TIMEOUT)
- `core/affect/affective_resonance.py:106` [affect] `brain.think` - brain.think(
- `core/agency/agency_core.py:417` [agency] `structured_brain.generate` - shard_res = await structured_brain.generate(prompt, context=context)
- `core/agency/autonomous_task_engine.py:962` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2593` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2627` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2711` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2820` [agency] `llm.think` - raw = await llm.think(
- `core/agency/latent_distiller.py:46` [agency] `brain.think` - summary = await brain.think(
- `core/agi/curiosity_explorer.py:349` [agi] `router.think` - router.think(prompt, priority=0.3, is_background=True,
- `core/agi/hierarchical_planner.py:215` [agi] `router.think` - router.think(prompt, priority=0.3, is_background=True,
- `core/agi/skill_synthesizer.py:173` [agi] `router.think` - router.think(prompt, priority=0.2, is_background=True,
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
- `core/actuators/web_actuators.py:99` [actuators] `urllib.parse.urlparse` - parsed = urllib.parse.urlparse(url)
- `core/adapters/api_adapter.py:105` [adapters] `aiohttp.ClientSession` - self._http_session = aiohttp.ClientSession(
- `core/adapters/api_adapter.py:106` [adapters] `aiohttp.TCPConnector` - connector=aiohttp.TCPConnector(limit=100, keepalive_timeout=60)
- `core/adapters/chrome_cdp_transport.py:30` [adapters] `RuntimeError` - raise RuntimeError("websocket-client is required for Chrome CDP control") from exc
- `core/adapters/chrome_cdp_transport.py:31` [adapters] `websocket.create_connection` - ws = websocket.create_connection(target_ws_url, timeout=timeout)
- `core/agency/tool_orchestrator.py:214` [agency] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/bus/sensory_gate.py:209` [bus] `urllib.parse.quote` - f"&search={urllib.parse.quote(query)}&limit=3&namespace=0&format=json"
- `core/capabilities/web_interlocutor.py:367` [capabilities] `urllib.parse.quote` - quoted = urllib.parse.quote(target_url, safe=":/?&=%#")
- `core/capabilities/web_interlocutor.py:3071` [capabilities] `urllib.parse.urlparse` - current_parts = urllib.parse.urlparse(current)
- `core/capabilities/web_interlocutor.py:3072` [capabilities] `urllib.parse.urlparse` - desired_parts = urllib.parse.urlparse(desired)
- `core/collective/belief_sync.py:201` [collective] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/collective/belief_sync.py:231` [collective] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/collective/belief_sync.py:288` [collective] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/collective/belief_sync.py:382` [collective] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/collective/swarm_protocol.py:26` [collective] `socket.gethostname` - self.node_id = socket.gethostname()
- `core/collective/swarm_protocol.py:66` [collective] `logger.warning` - logger.warning("🕸️ Mycelial Swarm running in offline-only mode; socket binding unavailable.")
- `core/collective/swarm_protocol.py:97` [collective] `logger.debug` - logger.debug("Swarm listener close timed out; abandoning socket.")
- `core/consciousness/heartbeat.py:185` [consciousness] `socket.socket` - with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
- `core/consciousness/heartbeat.py:193` [consciousness] `logger.debug` - logger.debug("Failed to emit keep-alive socket message to watchdog: %s", e)
- `core/embodiment/iot_bridge.py:193` [embodiment] `urllib.parse.urlparse` - parsed = urllib.parse.urlparse(self.base)
- `core/embodiment/mock_iot_plug.py:32` [embodiment] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/embodiment/mock_iot_plug.py:58` [embodiment] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/embodiment/mock_iot_plug.py:90` [embodiment] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/embodiment/unity_bridge.py:30` [embodiment] `websockets.connect` - self.ws = await websockets.connect(uri)
- `core/epistemics/epistemic_reach.py:204` [epistemics] `join` - query = urllib.parse.quote(" ".join(terms[:6]))

## Degradation Handling

- Total `record_degradation()` calls: 3587
- Log-and-limp candidates: 3237
- Nearby fail-closed candidates: 350

Top limp-on files:

- `core/brain/llm/context_assembler.py`: 33
- `core/brain/inference_gate.py`: 30
- `core/consciousness/consciousness_bridge.py`: 29
- `core/brain/cognitive_engine.py`: 27
- `core/memory/memory_facade.py`: 27
- `core/senses/voice_engine.py`: 27
- `core/memory/episodic_memory.py`: 24
- `core/runtime/runtime_hygiene.py`: 24
- `core/capabilities/__init__.py`: 23
- `core/resilience/memory_governor.py`: 23

## Non-Runtime Candidates

- `core/architect/proof_obligations.py`
- `core/autonomy/autonomous_research_orchestrator.py`
- `core/autonomy/research_cycle.py`
- `core/autonomy/research_goal_filter.py`
- `core/autonomy/research_triggers.py`
- `core/brain/narrative_memory.py`
- `core/consciousness/animal_cognition.py`
- `core/consciousness/narrative_gravity.py`
- `core/consciousness/oscillatory_binding.py`
- `core/environment/experimentation.py`
- `core/evaluation/behavioral_proof.py`
- `core/evaluation/proof_acceptance.py`
- `core/factory/repo_cartographer.py`
- `core/identity/narrative_thread.py`
- `core/lab/experiment_designer.py`
- `core/lab/research_lab.py`
- `core/lab/research_memory.py`
- `core/learning/proof_obligations.py`
- `core/memory/hippocampus.py`
- `core/reasoning/proof_answer_solver.py`
- `core/reproducibility/proof_substrate.py`
- `core/runtime/proof_kernel_bridge.py`
- `core/runtime/proof_policy.py`
- `core/search/research_pipeline.py`
- `core/skills/deep_research.py`

## Consolidation Candidates

- `core/audits/`: 2 file(s), 267 line(s)
- `core/coherence/`: 2 file(s), 407 line(s)
- `core/consent/`: 2 file(s), 167 line(s)
- `core/constitution/`: 1 file(s), 25 line(s)
- `core/creativity/`: 2 file(s), 801 line(s)
- `core/ethics/`: 2 file(s), 580 line(s)
- `core/evals/`: 2 file(s), 293 line(s)
- `core/intent/`: 2 file(s), 688 line(s)
- `core/latent/`: 1 file(s), 56 line(s)
- `core/maintenance/`: 2 file(s), 295 line(s)
- `core/media/`: 2 file(s), 336 line(s)
- `core/middleware/`: 2 file(s), 254 line(s)
- `core/multimodal/`: 2 file(s), 185 line(s)
- `core/ontology/`: 2 file(s), 169 line(s)
- `core/persistence/`: 2 file(s), 618 line(s)
- `core/play/`: 1 file(s), 228 line(s)
- `core/predictive/`: 2 file(s), 186 line(s)
- `core/reproducibility/`: 2 file(s), 497 line(s)
- `core/science/`: 1 file(s), 139 line(s)
- `core/search/`: 2 file(s), 1758 line(s)
- `core/sensors/`: 1 file(s), 159 line(s)
- `core/services/`: 2 file(s), 31 line(s)
- `core/skill_management/`: 1 file(s), 367 line(s)
- `core/transparency/`: 2 file(s), 317 line(s)
- `core/twins/`: 1 file(s), 97 line(s)
