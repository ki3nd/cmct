from cmct.prompts import PROMPTS, prompts_for
from tests.conftest import load_fixture

# cmct's dataset name -> the spelling the frozen prompt fixture is keyed by.
PROMPT_FIXTURE_KEYS = {
    "officehome": "office_home",
    "visda17": "visda",
    "digits": "digits",
    "office31": "office31",
    "domainnet": "domain_net",
    "imageclef": "image_clef",
}


def test_prompt_literals_match_the_frozen_baseline():
    """The prompt lists are not derived from the dataset -- nothing in the
    code makes them agree with the class names. If they ever drifted, the
    branch_mlp cosine logits would be silently mislabelled, so pin them
    against a frozen baseline (tests/fixtures/prompts.json; see
    tests/fixtures/README.md) rather than trusting the literals to stay in
    sync with themselves."""
    baseline = load_fixture("prompts.json")
    assert set(baseline) == set(PROMPT_FIXTURE_KEYS.values())
    for name, fixture_key in PROMPT_FIXTURE_KEYS.items():
        assert prompts_for(name) == baseline[fixture_key], name
    assert set(PROMPTS) == set(PROMPT_FIXTURE_KEYS)


def test_both_branches_are_handed_the_same_prompt_list():
    """Neither model type builds prompts of its own any more -- both take a
    ready list, so train.py is the only place a list is chosen. This pins the
    signatures; that train.py hands BOTH the same list is not checkable here
    and is enforced by there being one `prompts` local (cmct/train.py)."""
    import inspect

    from cmct.branch_lora.model import LoraCLIP
    from cmct.branch_mlp.backbone import ClipBackbone

    for cls in (LoraCLIP, ClipBackbone):
        params = list(inspect.signature(cls.__init__).parameters)
        assert params[1] == "prompts", cls.__name__
        assert "template" not in params, cls.__name__


# Office-31's real class directories, in the case-sensitive ASCII order dassl
# discovers them. Unlike office_home.py and domainnet.py, dassl's office31.py
# passes these through WITHOUT .lower() (vendor/dassl/.../office31.py:59), so
# they reach the alignment guard with their own spelling.
OFFICE31_CLASSNAMES = [
    "back_pack", "bike", "bike_helmet", "bookcase", "bottle", "calculator",
    "desk_chair", "desk_lamp", "desktop_computer", "file_cabinet", "headphones",
    "keyboard", "laptop_computer", "letter_tray", "mobile_phone", "monitor",
    "mouse", "mug", "paper_notebook", "pen", "phone", "printer", "projector",
    "punchers", "ring_binder", "ruler", "scissors", "speaker", "stapler",
    "tape_dispenser", "trash_can",
]


def test_office31_directory_spellings_pass_the_alignment_guard():
    # "back_pack" vs the prompt's "backpack" is the case that matters: a plain
    # underscores-to-spaces comparison rejects it, which would abort every
    # office31 run at startup.
    from cmct.train import misaligned_prompts

    assert misaligned_prompts(prompts_for("office31"), OFFICE31_CLASSNAMES) == []


def test_the_alignment_guard_still_catches_a_reordered_list():
    from cmct.train import misaligned_prompts

    swapped = list(OFFICE31_CLASSNAMES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert len(misaligned_prompts(prompts_for("office31"), swapped)) == 2
