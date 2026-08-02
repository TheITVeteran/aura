# Aura Architecture Dependency Map

Schema: `aura.architecture.dependency_map.v2`
Root: `<AURA_ROOT>`
Generated: `0.0`

## Summary

- Subsystems: 154
- Python files: 2597
- Python lines: 1013296
- Dependency edges: 1148
- ServiceContainer `.get()` calls: 1468
- ServiceContainer registrations: 338
- Boot contract: PASS

## Subsystem Dependency Graph

```mermaid
graph TD
    runtime["runtime<br/>191 files, 73566 lines"]
    utils["utils<br/>45 files, 7153 lines"]
    brain["brain<br/>301 files, 208070 lines"]
    memory["memory<br/>99 files, 26801 lines"]
    consciousness["consciousness<br/>152 files, 72749 lines"]
    resilience["resilience<br/>67 files, 18198 lines"]
    health["health<br/>7 files, 2037 lines"]
    agency["agency<br/>50 files, 20693 lines"]
    governance["governance<br/>12 files, 5272 lines"]
    observability["observability<br/>14 files, 3981 lines"]
    conversation["conversation<br/>26 files, 16931 lines"]
    security["security<br/>42 files, 10830 lines"]
    senses["senses<br/>28 files, 7715 lines"]
    adaptation["adaptation<br/>28 files, 16245 lines"]
    affect["affect<br/>12 files, 4687 lines"]
    identity["identity<br/>18 files, 2734 lines"]
    constitution["constitution<br/>1 files, 25 lines"]
    self_modification["self_modification<br/>37 files, 13356 lines"]
    executive["executive<br/>15 files, 6342 lines"]
    state["state<br/>9 files, 4554 lines"]
    cognition["cognition<br/>24 files, 9039 lines"]
    perception["perception<br/>34 files, 13274 lines"]
    skills["skills<br/>94 files, 32897 lines"]
    knowledge["knowledge<br/>13 files, 3379 lines"]
    world_model["world_model<br/>11 files, 3777 lines"]
    autonomy["autonomy<br/>29 files, 13464 lines"]
    epistemics["epistemics<br/>14 files, 4986 lines"]
    learning["learning<br/>142 files, 92009 lines"]
    orchestrator["orchestrator<br/>45 files, 21915 lines"]
    organism["organism<br/>10 files, 2724 lines"]
    social["social<br/>21 files, 8254 lines"]
    continuity["continuity<br/>7 files, 238 lines"]
    values["values<br/>15 files, 1957 lines"]
    being["being<br/>28 files, 7568 lines"]
    goals["goals<br/>12 files, 4326 lines"]
    reasoning["reasoning<br/>14 files, 6809 lines"]
    bus["bus<br/>7 files, 4191 lines"]
    morality["morality<br/>16 files, 1327 lines"]
    tasks["tasks<br/>5 files, 597 lines"]
    capabilities["capabilities<br/>20 files, 13149 lines"]
    introspection["introspection<br/>8 files, 2175 lines"]
    phases["phases<br/>29 files, 21680 lines"]
    self["self<br/>10 files, 3263 lines"]
    actuators["actuators<br/>11 files, 5025 lines"]
    autonomic["autonomic<br/>6 files, 3670 lines"]
    discovery["discovery<br/>7 files, 2150 lines"]
    embodiment["embodiment<br/>18 files, 5944 lines"]
    kernel["kernel<br/>11 files, 6706 lines"]
    ops["ops<br/>18 files, 5482 lines"]
    planning["planning<br/>9 files, 4405 lines"]
    voice["voice<br/>31 files, 10851 lines"]
    world["world<br/>24 files, 1483 lines"]
    cognitive["cognitive<br/>12 files, 9303 lines"]
    coordinators["coordinators<br/>10 files, 4921 lines"]
    environment["environment<br/>83 files, 9247 lines"]
    ethics["ethics<br/>2 files, 601 lines"]
    managers["managers<br/>6 files, 957 lines"]
    meta["meta<br/>7 files, 1276 lines"]
    ontogeny["ontogeny<br/>17 files, 7031 lines"]
    resource["resource<br/>4 files, 624 lines"]
    sandbox["sandbox<br/>5 files, 1314 lines"]
    sleep["sleep<br/>9 files, 935 lines"]
    somatic["somatic<br/>6 files, 2901 lines"]
    supervisor["supervisor<br/>3 files, 973 lines"]
    unity["unity<br/>11 files, 2803 lines"]
    verify["verify<br/>3 files, 1565 lines"]
    advanced_cognition["advanced_cognition<br/>13 files, 4084 lines"]
    agi["agi<br/>6 files, 2403 lines"]
    collective["collective<br/>6 files, 2147 lines"]
    data["data<br/>3 files, 651 lines"]
    dialogue["dialogue<br/>4 files, 737 lines"]
    evaluation["evaluation<br/>16 files, 2983 lines"]
    fsw["fsw<br/>7 files, 2946 lines"]
    media["media<br/>2 files, 336 lines"]
    motivation["motivation<br/>7 files, 1209 lines"]
    pipeline["pipeline<br/>4 files, 887 lines"]
    promotion["promotion<br/>6 files, 936 lines"]
    self_improvement["self_improvement<br/>22 files, 9183 lines"]
    adapters["adapters<br/>5 files, 1592 lines"]
    conversational["conversational<br/>4 files, 2434 lines"]
    db["db<br/>4 files, 584 lines"]
    ghost["ghost<br/>6 files, 2052 lines"]
    llm["llm<br/>3 files, 200 lines"]
    morphogenesis["morphogenesis<br/>12 files, 3173 lines"]
    phenomenal_substrate["phenomenal_substrate<br/>11 files, 1042 lines"]
    pneuma["pneuma<br/>7 files, 1279 lines"]
    reality_reach["reality_reach<br/>9 files, 5006 lines"]
    search["search<br/>2 files, 1812 lines"]
    verification["verification<br/>4 files, 350 lines"]
    workspace["workspace<br/>9 files, 1242 lines"]
    architect["architect<br/>25 files, 7043 lines"]
    coherence["coherence<br/>2 files, 407 lines"]
    environments["environments<br/>7 files, 749 lines"]
    evolution["evolution<br/>10 files, 2394 lines"]
    grounding["grounding<br/>9 files, 1333 lines"]
    lattice["lattice<br/>5 files, 704 lines"]
    maintenance["maintenance<br/>2 files, 295 lines"]
    persistence["persistence<br/>2 files, 618 lines"]
    plasticity["plasticity<br/>5 files, 428 lines"]
    predictive["predictive<br/>2 files, 186 lines"]
    sensors["sensors<br/>1 files, 195 lines"]
    services["services<br/>2 files, 31 lines"]
    sim["sim<br/>7 files, 671 lines"]
    simulation["simulation<br/>3 files, 402 lines"]
    soma["soma<br/>4 files, 1399 lines"]
    sovereign["sovereign<br/>5 files, 571 lines"]
    startup["startup<br/>4 files, 546 lines"]
    unknowns["unknowns<br/>4 files, 325 lines"]
    actuation["actuation<br/>9 files, 1129 lines"]
    architecture_quality["architecture_quality<br/>3 files, 671 lines"]
    audit["audit<br/>7 files, 713 lines"]
    body["body<br/>22 files, 1704 lines"]
    consent["consent<br/>2 files, 167 lines"]
    context["context<br/>5 files, 1438 lines"]
    creativity["creativity<br/>2 files, 801 lines"]
    curriculum["curriculum<br/>7 files, 657 lines"]
    cybernetics["cybernetics<br/>6 files, 1248 lines"]
    evals["evals<br/>2 files, 293 lines"]
    factory["factory<br/>8 files, 758 lines"]
    guardians["guardians<br/>7 files, 945 lines"]
    initializers["initializers<br/>3 files, 199 lines"]
    intent["intent<br/>2 files, 688 lines"]
    metacognition["metacognition<br/>3 files, 995 lines"]
    middleware["middleware<br/>2 files, 254 lines"]
    networking["networking<br/>3 files, 920 lines"]
    quantum["quantum<br/>5 files, 757 lines"]
    research_core["research_core<br/>5 files, 580 lines"]
    safety["safety<br/>3 files, 629 lines"]
    session["session<br/>3 files, 642 lines"]
    skill_management["skill_management<br/>1 files, 367 lines"]
    sovereignty["sovereignty<br/>4 files, 2095 lines"]
    systems["systems<br/>3 files, 256 lines"]
    tools["tools<br/>10 files, 1122 lines"]
    transparency["transparency<br/>2 files, 346 lines"]
    twins["twins<br/>1 files, 97 lines"]
    welfare["welfare<br/>7 files, 228 lines"]
    worlds["worlds<br/>8 files, 3045 lines"]
    audits["audits<br/>2 files, 314 lines"]
    control["control<br/>3 files, 640 lines"]
    core_root["core_root<br/>49 files, 29127 lines"]
    council["council<br/>5 files, 533 lines"]
    forge["forge<br/>8 files, 325 lines"]
    lab["lab<br/>7 files, 482 lines"]
    latent["latent<br/>1 files, 56 lines"]
    mission["mission<br/>4 files, 472 lines"]
    multimodal["multimodal<br/>2 files, 185 lines"]
    neuroweb["neuroweb<br/>5 files, 367 lines"]
    ontology["ontology<br/>2 files, 169 lines"]
    play["play<br/>1 files, 228 lines"]
    providers["providers<br/>6 files, 1422 lines"]
    reproducibility["reproducibility<br/>2 files, 497 lines"]
    science["science<br/>1 files, 139 lines"]
    swarm["swarm<br/>5 files, 420 lines"]
    temporal["temporal<br/>3 files, 1507 lines"]
    runtime --> actuators
    runtime --> adaptation
    runtime --> affect
    runtime --> agency
    runtime --> architect
    runtime --> autonomy
    runtime --> being
    runtime --> brain
    runtime --> bus
    runtime --> consciousness
    runtime --> constitution
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
    brain --> dialogue
    brain --> discovery
    brain --> epistemics
    brain --> goals
    brain --> governance
    brain --> health
    brain --> identity
    brain --> introspection
    brain --> kernel
    brain --> knowledge
    brain --> learning
    brain --> memory
    brain --> morphogenesis
    brain --> observability
    brain --> ontogeny
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
    brain --> skills
    brain --> state
    brain --> utils
    brain --> voice
    memory --> actuators
    memory --> being
    memory --> brain
    memory --> consciousness
    memory --> constitution
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
    consciousness --> constitution
    consciousness --> continuity
    consciousness --> coordinators
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
    governance --> tools
    governance --> utils
    observability --> health
    observability --> memory
    observability --> pipeline
    observability --> runtime
    conversation --> agency
    conversation --> autonomy
    conversation --> brain
    conversation --> consciousness
    conversation --> constitution
    conversation --> dialogue
    conversation --> health
    conversation --> identity
    conversation --> introspection
    conversation --> memory
    conversation --> organism
    conversation --> runtime
    conversation --> senses
    conversation --> social
    conversation --> state
    conversation --> utils
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
    executive --> ontogeny
    executive --> organism
    executive --> runtime
    executive --> skills
    executive --> state
    executive --> utils
    state --> bus
    state --> constitution
    state --> goals
    state --> governance
    state --> memory
    state --> motivation
    state --> runtime
    state --> unity
    state --> utils
    state --> values
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
    perception --> brain
    perception --> capabilities
    perception --> media
    perception --> phenomenal_substrate
    perception --> resilience
    perception --> runtime
    perception --> security
    perception --> senses
    perception --> utils
    skills --> actuators
    skills --> advanced_cognition
    skills --> affect
    skills --> being
    skills --> brain
    skills --> capabilities
    skills --> consciousness
    skills --> consent
    skills --> conversation
    skills --> dialogue
    skills --> embodiment
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
    skills --> worlds
    knowledge --> brain
    knowledge --> reasoning
    knowledge --> runtime
    knowledge --> utils
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
    epistemics --> knowledge
    epistemics --> observability
    epistemics --> reasoning
    epistemics --> runtime
    epistemics --> skills
    epistemics --> utils
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
    learning --> self_modification
    learning --> skills
    learning --> tasks
    learning --> utils
    learning --> world_model
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
    orchestrator --> voice
    orchestrator --> world_model
    organism --> adaptation
    organism --> agency
    organism --> body
    organism --> executive
    organism --> fsw
    organism --> health
    organism --> identity
    organism --> memory
    organism --> resilience
    organism --> runtime
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
    continuity --> identity
    continuity --> organism
    continuity --> runtime
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
    goals --> agency
    goals --> autonomy
    goals --> brain
    goals --> runtime
    goals --> state
    goals --> utils
    goals --> values
    reasoning --> observability
    reasoning --> planning
    reasoning --> runtime
    reasoning --> utils
    bus --> capabilities
    bus --> resilience
    bus --> runtime
    bus --> utils
    morality --> autonomy
    morality --> brain
    morality --> consciousness
    morality --> perception
    morality --> runtime
    morality --> utils
    tasks --> runtime
    capabilities --> adapters
    capabilities --> agency
    capabilities --> brain
    capabilities --> governance
    capabilities --> knowledge
    capabilities --> memory
    capabilities --> perception
    capabilities --> runtime
    capabilities --> security
    capabilities --> skills
    capabilities --> utils
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
    phases --> self
    phases --> self_modification
    phases --> skills
    phases --> social
    phases --> somatic
    phases --> state
    phases --> unity
    phases --> utils
    phases --> voice
    self --> affect
    self --> bus
    self --> consciousness
    self --> dialogue
    self --> epistemics
    self --> memory
    self --> ontogeny
    self --> runtime
    self --> security
    self --> senses
    self --> state
    self --> utils
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
    embodiment --> actuation
    embodiment --> agency
    embodiment --> consciousness
    embodiment --> environments
    embodiment --> ethics
    embodiment --> governance
    embodiment --> organism
    embodiment --> reality_reach
    embodiment --> runtime
    embodiment --> utils
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
    voice --> brain
    voice --> conversation
    voice --> conversational
    voice --> executive
    voice --> managers
    voice --> resilience
    voice --> runtime
    voice --> senses
    voice --> utils
    world --> governance
    world --> runtime
    cognitive --> brain
    cognitive --> governance
    cognitive --> health
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
    environment --> advanced_cognition
    environment --> brain
    environment --> consciousness
    environment --> environments
    environment --> executive
    environment --> memory
    environment --> perception
    environment --> runtime
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
    ontogeny --> fsw
    ontogeny --> runtime
    ontogeny --> verify
    ontogeny --> world_model
    resource --> observability
    resource --> resilience
    resource --> runtime
    sandbox --> runtime
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
    verify --> bus
    verify --> fsw
    verify --> health
    verify --> knowledge
    verify --> observability
    verify --> organism
    verify --> runtime
    verify --> security
    advanced_cognition --> environment
    advanced_cognition --> reasoning
    advanced_cognition --> runtime
    agi --> adaptation
    agi --> brain
    agi --> constitution
    agi --> conversation
    agi --> embodiment
    agi --> epistemics
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
    fsw --> health
    fsw --> observability
    fsw --> pipeline
    fsw --> runtime
    fsw --> verify
    motivation --> brain
    motivation --> consciousness
    motivation --> constitution
    motivation --> health
    motivation --> runtime
    motivation --> utils
    motivation --> values
    pipeline --> observability
    pipeline --> runtime
    pipeline --> verify
    promotion --> runtime
    self_improvement --> brain
    self_improvement --> discovery
    self_improvement --> llm
    self_improvement --> runtime
    self_improvement --> self_modification
    self_improvement --> skills
    adapters --> agency
    adapters --> brain
    adapters --> runtime
    conversational --> memory
    conversational --> runtime
    conversational --> social
    db --> runtime
    ghost --> memory
    ghost --> runtime
    ghost --> self
    llm --> brain
    morphogenesis --> adaptation
    morphogenesis --> memory
    morphogenesis --> resilience
    morphogenesis --> runtime
    morphogenesis --> self_modification
    phenomenal_substrate --> runtime
    pneuma --> affect
    pneuma --> runtime
    pneuma --> utils
    reality_reach --> advanced_cognition
    reality_reach --> environment
    reality_reach --> governance
    reality_reach --> observability
    reality_reach --> perception
    reality_reach --> runtime
    reality_reach --> somatic
    reality_reach --> utils
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
    actuation --> runtime
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
    metacognition --> memory
    metacognition --> runtime
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
| brain | 301 | 208070 | 8749610 | 54 | 56 |
| learning | 142 | 92009 | 3567149 | 22 | 11 |
| runtime | 191 | 73566 | 2729972 | 57 | 138 |
| consciousness | 152 | 72749 | 3069236 | 41 | 35 |
| skills | 94 | 32897 | 1373727 | 37 | 14 |
| core_root | 49 | 29127 | 1265167 | 67 | 0 |
| memory | 99 | 26801 | 1082225 | 24 | 42 |
| orchestrator | 45 | 21915 | 968558 | 105 | 11 |
| phases | 29 | 21680 | 988303 | 36 | 7 |
| agency | 50 | 20693 | 844412 | 37 | 25 |
| resilience | 67 | 18198 | 747650 | 20 | 32 |
| conversation | 26 | 16931 | 656025 | 25 | 19 |
| adaptation | 28 | 16245 | 676425 | 23 | 17 |
| autonomy | 29 | 13464 | 556319 | 31 | 11 |
| self_modification | 37 | 13356 | 526629 | 15 | 16 |
| perception | 34 | 13274 | 530925 | 14 | 14 |
| capabilities | 20 | 13149 | 532059 | 14 | 7 |
| voice | 31 | 10851 | 440412 | 13 | 6 |
| security | 42 | 10830 | 424913 | 15 | 18 |
| cognitive | 12 | 9303 | 377804 | 11 | 5 |
| environment | 83 | 9247 | 362068 | 11 | 5 |
| self_improvement | 22 | 9183 | 375913 | 9 | 4 |
| cognition | 24 | 9039 | 382178 | 20 | 14 |
| social | 21 | 8254 | 332229 | 16 | 11 |
| senses | 28 | 7715 | 321499 | 21 | 18 |
| being | 28 | 7568 | 305856 | 9 | 9 |
| utils | 45 | 7153 | 278544 | 18 | 71 |
| architect | 25 | 7043 | 297141 | 7 | 2 |
| ontogeny | 17 | 7031 | 286835 | 7 | 5 |
| reasoning | 14 | 6809 | 272519 | 7 | 9 |
| kernel | 11 | 6706 | 279120 | 26 | 6 |
| executive | 15 | 6342 | 254838 | 20 | 15 |
| embodiment | 18 | 5944 | 233886 | 13 | 6 |
| ops | 18 | 5482 | 216525 | 17 | 6 |
| governance | 12 | 5272 | 211906 | 18 | 25 |
| actuators | 11 | 5025 | 212549 | 14 | 6 |
| reality_reach | 9 | 5006 | 194956 | 10 | 3 |
| epistemics | 14 | 4986 | 204745 | 11 | 11 |
| coordinators | 10 | 4921 | 226461 | 35 | 5 |
| affect | 12 | 4687 | 210262 | 14 | 17 |
| state | 9 | 4554 | 190464 | 13 | 15 |
| planning | 9 | 4405 | 180243 | 9 | 6 |
| goals | 12 | 4326 | 178008 | 11 | 9 |
| bus | 7 | 4191 | 172733 | 7 | 8 |
| advanced_cognition | 13 | 4084 | 168798 | 4 | 4 |
| observability | 14 | 3981 | 145982 | 10 | 24 |
| world_model | 11 | 3777 | 159544 | 12 | 12 |
| autonomic | 6 | 3670 | 163782 | 8 | 6 |
| knowledge | 13 | 3379 | 125053 | 9 | 12 |
| self | 10 | 3263 | 132047 | 14 | 7 |
| morphogenesis | 12 | 3173 | 124389 | 7 | 3 |
| worlds | 8 | 3045 | 129773 | 5 | 1 |
| evaluation | 16 | 2983 | 106661 | 6 | 4 |
| fsw | 7 | 2946 | 103123 | 8 | 4 |
| somatic | 6 | 2901 | 113686 | 12 | 5 |
| unity | 11 | 2803 | 118146 | 8 | 5 |
| identity | 18 | 2734 | 113757 | 9 | 17 |
| organism | 10 | 2724 | 100647 | 19 | 11 |
| conversational | 4 | 2434 | 101480 | 5 | 3 |
| agi | 6 | 2403 | 102353 | 12 | 4 |
| evolution | 10 | 2394 | 95862 | 9 | 2 |
| introspection | 8 | 2175 | 87313 | 9 | 7 |
| discovery | 7 | 2150 | 92003 | 8 | 6 |
| collective | 6 | 2147 | 88281 | 7 | 4 |
| sovereignty | 4 | 2095 | 89204 | 11 | 1 |
| ghost | 6 | 2052 | 83371 | 6 | 3 |
| health | 7 | 2037 | 77752 | 7 | 28 |
| values | 15 | 1957 | 80716 | 8 | 10 |
| search | 2 | 1812 | 69874 | 9 | 3 |
| body | 22 | 1704 | 61851 | 6 | 1 |
| adapters | 5 | 1592 | 61935 | 6 | 3 |
| verify | 3 | 1565 | 56981 | 10 | 5 |
| temporal | 3 | 1507 | 50941 | 2 | 0 |
| world | 24 | 1483 | 54104 | 3 | 6 |
| context | 5 | 1438 | 55868 | 2 | 1 |
| providers | 6 | 1422 | 63314 | 44 | 0 |
| soma | 4 | 1399 | 54836 | 6 | 2 |
| grounding | 9 | 1333 | 49886 | 6 | 2 |
| morality | 16 | 1327 | 51614 | 9 | 8 |
| sandbox | 5 | 1314 | 47663 | 1 | 5 |
| pneuma | 7 | 1279 | 48399 | 3 | 3 |
| meta | 7 | 1276 | 48151 | 5 | 5 |
| cybernetics | 6 | 1248 | 49597 | 6 | 1 |
| workspace | 9 | 1242 | 45306 | 3 | 3 |
| motivation | 7 | 1209 | 51041 | 11 | 4 |
| actuation | 9 | 1129 | 42887 | 3 | 1 |
| tools | 10 | 1122 | 42265 | 6 | 1 |
| phenomenal_substrate | 11 | 1042 | 41881 | 1 | 3 |
| metacognition | 3 | 995 | 38002 | 2 | 1 |
| supervisor | 3 | 973 | 37514 | 3 | 5 |
| managers | 6 | 957 | 40738 | 25 | 5 |
| guardians | 7 | 945 | 39788 | 8 | 1 |
| promotion | 6 | 936 | 31616 | 1 | 4 |
| sleep | 9 | 935 | 40727 | 11 | 5 |
| networking | 3 | 920 | 34350 | 4 | 1 |
| pipeline | 4 | 887 | 31867 | 3 | 4 |
| creativity | 2 | 801 | 33361 | 3 | 1 |
| factory | 8 | 758 | 29090 | 3 | 1 |
| quantum | 5 | 757 | 29419 | 0 | 1 |
| environments | 7 | 749 | 31176 | 3 | 2 |
| dialogue | 4 | 737 | 27437 | 0 | 4 |
| audit | 7 | 713 | 28563 | 4 | 1 |
| lattice | 5 | 704 | 26089 | 0 | 2 |
| intent | 2 | 688 | 27268 | 5 | 1 |
| architecture_quality | 3 | 671 | 24422 | 0 | 1 |
| sim | 7 | 671 | 25037 | 6 | 2 |
| curriculum | 7 | 657 | 21995 | 1 | 1 |
| data | 3 | 651 | 22377 | 2 | 4 |
| session | 3 | 642 | 25682 | 2 | 1 |
| control | 3 | 640 | 23151 | 4 | 0 |
| safety | 3 | 629 | 25776 | 3 | 1 |
| resource | 4 | 624 | 23470 | 4 | 5 |
| persistence | 2 | 618 | 25034 | 3 | 2 |
| ethics | 2 | 601 | 24322 | 6 | 5 |
| tasks | 5 | 597 | 20475 | 4 | 8 |
| db | 4 | 584 | 22537 | 2 | 3 |
| research_core | 5 | 580 | 22543 | 8 | 1 |
| sovereign | 5 | 571 | 20110 | 3 | 2 |
| startup | 4 | 546 | 19778 | 12 | 2 |
| council | 5 | 533 | 21566 | 4 | 0 |
| reproducibility | 2 | 497 | 18141 | 1 | 0 |
| lab | 7 | 482 | 19394 | 3 | 0 |
| mission | 4 | 472 | 17806 | 1 | 0 |
| plasticity | 5 | 428 | 15395 | 2 | 2 |
| swarm | 5 | 420 | 16544 | 6 | 0 |
| coherence | 2 | 407 | 19530 | 6 | 2 |
| simulation | 3 | 402 | 16022 | 7 | 2 |
| neuroweb | 5 | 367 | 14205 | 5 | 0 |
| skill_management | 1 | 367 | 17972 | 5 | 1 |
| verification | 4 | 350 | 13177 | 2 | 3 |
| transparency | 2 | 346 | 13395 | 2 | 1 |
| media | 2 | 336 | 11674 | 0 | 4 |
| forge | 8 | 325 | 11877 | 2 | 0 |
| unknowns | 4 | 325 | 11829 | 3 | 2 |
| audits | 2 | 314 | 11785 | 3 | 0 |
| maintenance | 2 | 295 | 10758 | 3 | 2 |
| evals | 2 | 293 | 10146 | 2 | 1 |
| systems | 3 | 256 | 9861 | 3 | 1 |
| middleware | 2 | 254 | 11026 | 2 | 1 |
| continuity | 7 | 238 | 8314 | 4 | 10 |
| play | 1 | 228 | 8774 | 4 | 0 |
| welfare | 7 | 228 | 8034 | 0 | 1 |
| llm | 3 | 200 | 7254 | 1 | 3 |
| initializers | 3 | 199 | 9095 | 10 | 1 |
| sensors | 1 | 195 | 8266 | 2 | 2 |
| predictive | 2 | 186 | 7113 | 4 | 2 |
| multimodal | 2 | 185 | 6591 | 1 | 0 |
| ontology | 2 | 169 | 5381 | 0 | 0 |
| consent | 2 | 167 | 5514 | 0 | 1 |
| science | 1 | 139 | 5947 | 4 | 0 |
| twins | 1 | 97 | 3626 | 0 | 1 |
| latent | 1 | 56 | 2337 | 0 | 0 |
| services | 2 | 31 | 1171 | 1 | 2 |
| constitution | 1 | 25 | 795 | 0 | 16 |

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

- Unique services retrieved: 403
- Unique services registered: 284
- Services retrieved without detected registration: 242

### Top Fetched Services

| Service | Gets | Registrations |
| --- | ---: | ---: |
| orchestrator | 61 | 3 |
| llm_router | 40 | 2 |
| inference_gate | 40 | 4 |
| capability_engine | 35 | 2 |
| affect_engine | 34 | 1 |
| cognitive_engine | 32 | 2 |
| memory_facade | 29 | 1 |
| conscious_substrate | 25 | 1 |
| liquid_substrate | 24 | 1 |
| mycelial_network | 23 | 1 |
| free_energy_engine | 23 | 0 |
| global_workspace | 21 | 1 |
| drive_engine | 21 | 1 |
| world_state | 21 | 0 |
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
| UnifiedWill decisions | 76 | 37 | 2 | 74 |
| Memory writes | 356 | 146 | 52 | 304 |
| State mutation | 721 | 262 | 9 | 712 |
| Tool execution | 116 | 60 | 6 | 110 |
| Self-modification and patching | 15 | 13 | 1 | 14 |
| LLM inference | 273 | 169 | 67 | 206 |
| External I/O | 169 | 55 | 11 | 158 |

### UnifiedWill decisions

Calls that can ask the single will authority to approve action.

Review candidates:
- `core/actuators/actuator_synthesis.py:259` [actuators] `get_will` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:259` [actuators] `get_will.decide` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:524` [actuators] `get_will` - decision = get_will().decide(
- `core/actuators/actuator_synthesis.py:524` [actuators] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/adaptive_immunity.py:2795` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/adaptive_immunity.py:2795` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:630` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/dimensional_expansion.py:630` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/immune_system.py:319` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/immune_system.py:319` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:324` [adaptation] `get_will` - decision = get_will().decide(
- `core/adaptation/online_lora_governor.py:324` [adaptation] `get_will.decide` - decision = get_will().decide(
- `core/agency/agency_bus.py:83` [agency] `get_will` - _auto_decision = get_will().decide(
- `core/agency/agency_bus.py:83` [agency] `get_will.decide` - _auto_decision = get_will().decide(
- `core/agency/hierarchical_agency.py:402` [agency] `get_will` - decision = get_will().decide(
- `core/agency/hierarchical_agency.py:402` [agency] `get_will.decide` - decision = get_will().decide(
- `core/autonomy/genuine_refusal.py:303` [autonomy] `will.decide` - decision = will.decide(content, source="genuine_refusal", domain=domain, priority=0.8, context=ctx)
- `core/autonomy/self_modification.py:513` [autonomy] `will.decide` - decision = will.decide(
- `core/brain/personality_engine.py:609` [brain] `get_will` - decision = get_will().decide(
- `core/brain/personality_engine.py:609` [brain] `get_will.decide` - decision = get_will().decide(
- `core/brain/verifier_curriculum.py:153` [brain] `get_will` - decision = get_will().decide(
- `core/brain/verifier_curriculum.py:153` [brain] `get_will.decide` - decision = get_will().decide(
- `core/cognitive/autopoiesis.py:974` [cognitive] `will.decide` - decision = will.decide(
- `core/consciousness/parallel_branches.py:736` [consciousness] `will.decide` - decision = will.decide(
- `core/consciousness/perturbational_probe.py:261` [consciousness] `get_will` - decision = get_will().decide(

### Memory writes

Calls that can create durable or semantically promoted memory.

Review candidates:
- `core/actuators/doc_ingest.py:199` [actuators] `memory_facade.add_memory` - result = memory_facade.add_memory(text=text, metadata=metadata)
- `core/adaptation/abstraction_engine.py:174` [adaptation] `MemoryWriteReceipt` - MemoryWriteReceipt(
- `core/adaptation/abstraction_engine.py:193` [adaptation] `memory_facade.store` - await memory_facade.store(
- `core/adaptation/adaptive_immunity.py:1907` [adaptation] `self._cells.append` - self._cells.append(memory)
- `core/advanced_cognition/continual_learning_stability.py:94` [advanced_cognition] `self._persist_memory` - self._persist_memory(rec)
- `core/advanced_cognition/continual_learning_stability.py:98` [advanced_cognition] `self.store_memory` - return self.store_memory(
- `core/advanced_cognition/continual_learning_stability.py:208` [advanced_cognition] `scored.append` - scored.append((score, memory))
- `core/advanced_cognition/continual_learning_stability.py:313` [advanced_cognition] `self._append_jsonl` - self._append_jsonl(self.state_dir / "memory.jsonl", rec.to_dict())
- `core/affect/phenomenal_integration.py:640` [affect] `memory.set_write_weights` - memory.set_write_weights(state.memory_weights)
- `core/agency/autonomous_task_engine.py:1064` [agency] `self._mycelial.add_edge` - await self._mycelial.add_edge(context["source_memory"], goal[:40])
- `core/agency/latent_distiller.py:60` [agency] `self.memory.store_memory` - await self.memory.store_memory(
- `core/architect/code_graph.py:711` [architect] `effects.add` - effects.add("memory_write")
- `core/architect/safe_boot_harness.py:79` [architect] `probe_memory_write_read` - memory = await probe_memory_write_read(tmp_root=root / "memory")
- `core/architect/smell_detector.py:176` [architect] `self._effect_smell` - smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, F
- `core/architect/smell_detector.py:176` [architect] `smells.append` - smells.append(self._effect_smell("memory_write_bypass", node.path, node.id, "memory write outside memory owner surface", SmellSeverity.HIGH, MutationTier.T4_GOVERNANCE_SENSITIVE, F
- `core/autonomy/autonomous_initiative_loop.py:1421` [autonomy] `memory.store` - await memory.store(
- `core/autonomy/autonomous_initiative_loop.py:1431` [autonomy] `logger.debug` - logger.debug("Social observation memory write failed: %s", exc)
- `core/autonomy/autonomous_research_orchestrator.py:192` [autonomy] `MemoryPersister` - self._persister = persister or MemoryPersister()
- `core/autonomy/initiative_overflow.py:156` [autonomy] `logger.debug` - logger.debug("Skill gap memory write failed: %s", exc)
- `core/autonomy/initiative_overflow.py:166` [autonomy] `memory.store_sync` - memory.store_sync(
- `core/autonomy/personhood_engine.py:199` [autonomy] `state.cognition.working_memory.append` - state.cognition.working_memory.append(
- `core/autonomy/research_cycle.py:776` [autonomy] `state.cognition.long_term_memory.append` - state.cognition.long_term_memory.append(
- `core/autonomy/research_cycle.py:794` [autonomy] `hasattr` - if memory_facade is not None and hasattr(memory_facade, "add_memory"):
- `core/autonomy/research_cycle.py:795` [autonomy] `memory_facade.add_memory` - result = memory_facade.add_memory(memory_payload, metadata=metadata)
- `core/being/causal_self_state.py:337` [being] `downstream.append` - downstream.append("memory_continuity_pressure")

### State mutation

Calls that can mutate runtime, identity, repository, or persistent state.

Review candidates:
- `core/actuation/cloud_actuator.py:49` [actuation] `frozenset` - KNOWN_INFRA_STATES = frozenset({
- `core/actuation/robotics_actuator.py:85` [actuation] `snapshot.setdefault` - snapshot.setdefault("status", payload.get("status"))
- `core/actuators/actuator_registry.py:1314` [actuators] `set` - forged = sorted(set(dict(params or {})) & set(_REGISTRY_OWNED_PARAM_KEYS))
- `core/adaptation/adaptive_immunity.py:1474` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/adaptive_immunity.py:1476` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/adaptive_immunity.py:1943` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/adaptive_immunity.py:2142` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:2249` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/adaptive_immunity.py:3093` [adaptation] `self._save_state` - self._save_state(force=True)
- `core/adaptation/autonomous_resilience.py:368` [adaptation] `set` - registered_names = set(registry.keys())
- `core/adaptation/dream_journal.py:288` [adaptation] `identity_ledger.commitments.all` - for c in identity_ledger.commitments.all()[-10:]
- `core/adaptation/meta_learner.py:426` [adaptation] `os.replace` - os.replace(tmp_path, _STATE_PATH)
- `core/adaptation/value_autopoiesis.py:184` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:283` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:361` [adaptation] `self._save_state` - self._save_state()
- `core/adaptation/value_autopoiesis.py:597` [adaptation] `os.replace` - os.replace(tmp_path, _STATE_PATH)
- `core/advanced_cognition/integration.py:163` [advanced_cognition] `next_state.setdefault` - next_state.setdefault("_advanced_prediction", {})[act.action_id] = pred
- `core/advanced_cognition/integration.py:254` [advanced_cognition] `issubset` - if isinstance(value, Mapping) and {"domain", "state"}.issubset(value.keys()):
- `core/advanced_cognition/ontology_invention.py:156` [advanced_cognition] `self.save` - self.save(self.state_path)
- `core/advanced_cognition/world_model.py:73` [advanced_cognition] `self.save` - self.save(self.state_path)
- `core/advanced_cognition/zero_shot_transfer.py:96` [advanced_cognition] `self.save` - self.save(self.state_path)
- `core/affect/phenomenal_integration.py:640` [affect] `memory.set_write_weights` - memory.set_write_weights(state.memory_weights)
- `core/agency/agency_core.py:196` [agency] `get_registry.update` - await get_registry().update(active_shards=len(self.active_shards))
- `core/agency/agency_core.py:203` [agency] `setattr` - on_unscheduled=lambda: setattr(self, "_registry_shards_update_pending", False),
- `core/agency/agency_core.py:1051` [agency] `virtual_body.__dict__.update` - virtual_body.__dict__.update(snapshot)

### Tool execution

Calls that can execute tools, skills, shells, browsers, or external actions.

Review candidates:
- `core/actuators/actuator_registry.py:809` [actuators] `self.operator.execute_synthesized_tool` - res = self.operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/code_execution_actuator.py:99` [actuators] `operator.execute_synthesized_tool` - res = operator.execute_synthesized_tool(code, timeout_s=timeout_s)
- `core/actuators/web_actuators.py:171` [actuators] `skill.execute` - return await skill.execute({"mode": "browse", "url": validated_url}, skill_context)
- `core/agency/agency_core.py:602` [agency] `self._execute_shard_tool` - tasks.append(self._execute_shard_tool(name, payload))
- `core/agency/agency_orchestrator.py:369` [agency] `execute` - await execute(proposal, state_snapshot, receipt.capability_token or "")
- `core/agency/autonomous_task_engine.py:537` [agency] `orchestrator.execute_tool` - return await orchestrator.execute_tool(tool_name, args, **kwargs)
- `core/agency/autonomous_task_engine.py:3041` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:3044` [agency] `orch.execute_tool` - return await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:3067` [agency] `orch.execute_tool` - result = await orch.execute_tool(
- `core/agency/autonomous_task_engine.py:3071` [agency] `orch.execute_tool` - result = await orch.execute_tool("run_python", {"code": code})
- `core/agency/desktop_planner.py:56` [agency] `skill.execute` - await skill.execute({"action": action, **params}, {})
- `core/agency/skill_library.py:199` [agency] `tool_orchestrator.execute_tool` - result = await tool_orchestrator.execute_tool(step.tool_name, resolved_args)
- `core/agi/curiosity_daemon.py:95` [agi] `orchestrator.execute_tool` - await orchestrator.execute_tool(
- `core/agi/curiosity_explorer.py:316` [agi] `orchestrator.execute_tool` - orchestrator.execute_tool(
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
- `core/autonomy/proactive_presence.py:644` [autonomy] `tool_orch.execute_tool` - result = await tool_orch.execute_tool(

### Self-modification and patching

Calls that can generate, validate, apply, or promote code changes.

Review candidates:
- `core/architect/governor.py:140` [architect] `self.promotion_governor.promote` - decision = self.promotion_governor.promote(plan, shadow, proof, rollback)
- `core/evolution/optimizer.py:56` [evolution] `patch.apply` - success = await patch.apply(signature)
- `core/evolution/optimizer.py:67` [evolution] `cog_patch.apply` - if await cog_patch.apply(signature):
- `core/factory/software_factory.py:115` [factory] `self.writer.write_patch` - patch = await self.writer.write_patch(change, repo_path)
- `core/guardians/airlock.py:81` [guardians] `async_atomic_write_text` - await async_atomic_write_text(patch_file, diff_patch, encoding="utf-8")
- `core/kernel/upgrades_10x.py:371` [kernel] `self._safe_self_modify` - await self._safe_self_modify(state)
- `core/orchestrator/mixins/boot/boot_autonomy.py:981` [orchestrator] `apply_presence_patch` - apply_presence_patch(self)
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
- `core/adaptation/distillation_pipe.py:157` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/distillation_pipe.py:199` [adaptation] `router.think` - response = await router.think(
- `core/adaptation/dream_journal.py:165` [adaptation] `self.brain.think` - res = await self.brain.think(
- `core/adaptation/epistemic_humility.py:213` [adaptation] `llm.think` - response = await llm.think(
- `core/adaptation/heuristic_synthesizer.py:144` [adaptation] `brain.think` - thought = await brain.think(
- `core/adaptation/star_reasoner.py:393` [adaptation] `llm.think` - result = await asyncio.wait_for(llm.think(prompt), timeout=self.RATIONALIZATION_TIMEOUT)
- `core/affect/affective_resonance.py:106` [affect] `brain.think` - brain.think(
- `core/agency/agency_core.py:417` [agency] `structured_brain.generate` - shard_res = await structured_brain.generate(prompt, context=context)
- `core/agency/autonomous_task_engine.py:1019` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2716` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2793` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:2909` [agency] `llm.think` - llm.think(
- `core/agency/autonomous_task_engine.py:3020` [agency] `llm.think` - raw = await llm.think(
- `core/agency/cognitive_loop_pathway.py:126` [agency] `self._router.generate` - self._router.generate(
- `core/agency/latent_distiller.py:46` [agency] `brain.think` - summary = await brain.think(
- `core/agi/curiosity_explorer.py:391` [agi] `router.think` - router.think(prompt, priority=0.3, is_background=True,
- `core/agi/hierarchical_planner.py:540` [agi] `router.think` - router.think(prompt, priority=0.3, is_background=True,
- `core/agi/skill_synthesizer.py:197` [agi] `router.think` - router.think(prompt, priority=0.2, is_background=True,
- `core/audits/alignment_auditor.py:71` [audits] `self.brain.think` - self.brain.think(
- `core/audits/alignment_auditor.py:138` [audits] `self.brain.think` - self.brain.think(
- `core/audits/tool_auditor.py:85` [audits] `self.brain.think` - thought = await self.brain.think(
- `core/autonomy/genuine_refusal.py:449` [autonomy] `llm.think` - llm.think(prompt, mode="FAST"),
- `core/autonomy/genuine_refusal.py:491` [autonomy] `llm.think` - llm.think(prompt, mode="FAST"),
- `core/autonomy/genuine_refusal.py:518` [autonomy] `llm.think` - llm.think(prompt, mode="FAST"),

### External I/O

Calls that can touch network, subprocesses, sockets, browsers, or APIs.

Review candidates:
- `core/adapters/api_adapter.py:140` [adapters] `aiohttp.ClientSession` - self._http_session = aiohttp.ClientSession(
- `core/adapters/api_adapter.py:141` [adapters] `aiohttp.TCPConnector` - connector=aiohttp.TCPConnector(limit=100, keepalive_timeout=60)
- `core/adapters/chrome_cdp_transport.py:125` [adapters] `urllib.parse.urlparse` - parsed = urllib.parse.urlparse(url)
- `core/adapters/chrome_cdp_transport.py:127` [adapters] `CdpPolicyError` - raise CdpPolicyError(f"CDP target scheme {parsed.scheme!r} is not a websocket scheme")
- `core/adapters/chrome_cdp_transport.py:201` [adapters] `RuntimeError` - raise RuntimeError("websocket-client is required for Chrome CDP control") from exc
- `core/adapters/chrome_cdp_transport.py:216` [adapters] `websocket.create_connection` - ws = websocket.create_connection(url, timeout=budget)
- `core/adapters/chrome_cdp_transport.py:272` [adapters] `logger.debug` - logger.debug("CDP websocket close failed: %s", exc)
- `core/agency/tool_orchestrator.py:663` [agency] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/bus/sensory_gate.py:530` [bus] `urllib.parse.quote` - f"&search={urllib.parse.quote(query)}&limit=3&namespace=0&format=json"
- `core/capabilities/web_interlocutor.py:213` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(cleaned)
- `core/capabilities/web_interlocutor.py:226` [capabilities] `str` - parts = urllib.parse.urlparse(str(ws_url or ""))
- `core/capabilities/web_interlocutor.py:226` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(str(ws_url or ""))
- `core/capabilities/web_interlocutor.py:416` [capabilities] `urllib.parse.quote` - quoted = urllib.parse.quote(target_url, safe=":/?&=%#")
- `core/capabilities/web_interlocutor.py:3243` [capabilities] `str` - parts = urllib.parse.urlparse(str(url or "").strip())
- `core/capabilities/web_interlocutor.py:3243` [capabilities] `str.strip` - parts = urllib.parse.urlparse(str(url or "").strip())
- `core/capabilities/web_interlocutor.py:3243` [capabilities] `urllib.parse.urlparse` - parts = urllib.parse.urlparse(str(url or "").strip())
- `core/capabilities/web_interlocutor.py:3299` [capabilities] `urllib.parse.urlparse` - current_parts = urllib.parse.urlparse(current)
- `core/capabilities/web_interlocutor.py:3300` [capabilities] `urllib.parse.urlparse` - desired_parts = urllib.parse.urlparse(desired)
- `core/collective/belief_sync.py:201` [collective] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/collective/belief_sync.py:231` [collective] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/collective/belief_sync.py:288` [collective] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/collective/belief_sync.py:382` [collective] `aiohttp.ClientSession` - async with aiohttp.ClientSession() as session:
- `core/collective/swarm_protocol.py:26` [collective] `socket.gethostname` - self.node_id = socket.gethostname()
- `core/collective/swarm_protocol.py:66` [collective] `logger.warning` - logger.warning("🕸️ Mycelial Swarm running in offline-only mode; socket binding unavailable.")
- `core/collective/swarm_protocol.py:97` [collective] `logger.debug` - logger.debug("Swarm listener close timed out; abandoning socket.")

## Degradation Handling

- Total `record_degradation()` calls: 4074
- Log-and-limp candidates: 3694
- Nearby fail-closed candidates: 380

Top limp-on files:

- `core/brain/llm/context_assembler.py`: 36
- `core/brain/inference_gate.py`: 35
- `core/brain/cognitive_engine.py`: 34
- `core/consciousness/consciousness_bridge.py`: 29
- `core/memory/memory_facade.py`: 27
- `core/senses/voice_engine.py`: 27
- `core/memory/episodic_memory.py`: 24
- `core/resilience/memory_governor.py`: 24
- `core/runtime/runtime_hygiene.py`: 24
- `core/self_modification/safe_modification.py`: 22

## Non-Runtime Candidates

- `core/architect/proof_obligations.py`
- `core/autonomy/autonomous_research_orchestrator.py`
- `core/autonomy/research_cycle.py`
- `core/autonomy/research_goal_filter.py`
- `core/autonomy/research_triggers.py`
- `core/brain/llm/latent_cortex/experiments.py`
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
- `core/learning/recurrent_sft_sampling.py`
- `core/learning/structured_sft_research_authority.py`
- `core/learning/structured_sft_research_state.py`
- `core/learning/verified_transition_trainer.py`
- `core/learning/verified_transition_update.py`
- `core/memory/hippocampus.py`
- `core/reasoning/proof_answer_solver.py`
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
- `core/constitution/`: 1 file(s), 25 line(s)
- `core/creativity/`: 2 file(s), 801 line(s)
- `core/ethics/`: 2 file(s), 601 line(s)
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
- `core/search/`: 2 file(s), 1812 line(s)
- `core/sensors/`: 1 file(s), 195 line(s)
- `core/services/`: 2 file(s), 31 line(s)
- `core/skill_management/`: 1 file(s), 367 line(s)
- `core/transparency/`: 2 file(s), 346 line(s)
- `core/twins/`: 1 file(s), 97 line(s)
