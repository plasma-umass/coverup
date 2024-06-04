import pytest
from hypothesis import given, strategies as st
from pathlib import Path
from coverup import utils
import subprocess

def test_format_ranges():
    assert "" == utils.format_ranges(set(), set())
    assert "1-5" == utils.format_ranges({1,2,3,5}, set())
    assert "1-3, 5" == utils.format_ranges({1,2,3,5}, {4})
    assert "1-6, 10-11, 30" == utils.format_ranges({1,2,4,5,6,10,11,30}, {8,14,15,16,17})


@given(st.integers(0, 10000), st.integers(1, 10000))
def test_format_ranges_is_sorted(a, b):
    b = a+b
    assert f"{a}-{b}" == utils.format_ranges({i for i in range(a,b+1)}, set())


def test_lines_branches_do():
    assert "line 123 does" == utils.lines_branches_do({123}, set(), set())
    assert "lines 123-125, 199 do" == utils.lines_branches_do({123,124,125,199}, {128}, set())
    assert "branch 1->5 does" == utils.lines_branches_do(set(), set(), {(1,5)})
    assert "branches 1->2, 1->5 do" == utils.lines_branches_do(set(), set(), {(1,5),(1,2)})
    assert "line 123 and branches 1->exit, 1->2 do" == utils.lines_branches_do({123}, set(), {(1,2),(1,0)})
    assert "lines 123-124 and branch 1->exit do" == utils.lines_branches_do({123, 124}, set(), {(1,0)})
    assert "lines 123-125 and branches 1->exit, 1->2 do" == utils.lines_branches_do({123,124,125}, set(), {(1,2),(1,0)})

    # if a line doesn't execute, neither do the branches that touch it...
    assert "lines 123-125 do" == utils.lines_branches_do({123,124,125}, set(), {(123,124), (10,125)})


@pytest.mark.parametrize('check', [False, True])
@pytest.mark.asyncio
async def test_subprocess_run(check):
    p = await utils.subprocess_run(['/bin/echo', 'hi!'], check=check)
    assert p.stdout == b"hi!\n"


@pytest.mark.asyncio
async def test_subprocess_run_fails_checked():
    with pytest.raises(subprocess.CalledProcessError) as e:
        await utils.subprocess_run(['/usr/bin/false'], check=True)

    assert e.value.stdout == b""


@pytest.mark.asyncio
async def test_subprocess_run_fails_not_checked():
    p = await utils.subprocess_run(['/usr/bin/false'])
    assert p.returncode != 0


@pytest.mark.asyncio
async def test_subprocess_run_timeout():
    with pytest.raises(subprocess.TimeoutExpired) as e:
        await utils.subprocess_run(['/bin/sleep', '2'], timeout=1)


def test_summary_coverage():
    import json

    cov = json.loads("""\
{
    "meta": {
        "software": "slipcover",
        "version": "1.0.14",
        "timestamp": "2024-06-04T13:41:57.105381",
        "branch_coverage": false,
        "show_contexts": false
    },
    "files": {
        "src/coverup/__init__.py": {
            "executed_lines": [ 1, 2 ],
            "missing_lines": [],
            "summary": {
                "covered_lines": 2,
                "missing_lines": 0,
                "percent_covered": 100.0
            }
        },
        "src/coverup/version.py": {
            "executed_lines": [ 1 ],
            "missing_lines": [],
            "summary": {
                "covered_lines": 1,
                "missing_lines": 0,
                "percent_covered": 100.0
            }
        },
        "src/coverup/coverup.py": {
            "executed_lines": [ 1, 2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21, 22,
                23, 24, 28, 29, 31, 33, 168, 174, 175, 192, 211, 212, 222, 273, 297, 298, 309,
                326, 331, 332, 333, 335, 347, 356, 363, 367, 372, 373, 375, 377, 378, 379, 380,
                381, 382, 385, 387, 390, 392, 395, 397, 398, 399, 400, 403, 405, 406, 408, 409,
                412, 414, 416, 417, 420, 425, 430, 431, 433, 434, 435, 436, 437, 438, 440, 441,
                443, 444, 446, 448, 449, 452, 470, 471, 472, 523, 530, 637, 644
            ],
            "missing_lines": [ 34, 36, 37, 38, 39, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 54,
                55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 67, 68, 70, 71, 72, 73, 75, 76, 78, 79,
                81, 82, 84, 85, 87, 88, 90, 91, 92, 94, 95, 96, 98, 99, 101, 102, 104, 105, 106,
                108, 109, 111, 112, 114, 115, 116, 118, 119, 120, 122, 123, 124, 126, 127, 128,
                130, 131, 132, 134, 135, 136, 138, 139, 140, 142, 143, 144, 146, 147, 148, 149,
                151, 152, 154, 155, 157, 159, 160, 162, 163, 165, 171, 180, 181, 182, 183, 184,
                185, 186, 187, 189, 196, 199, 200, 201, 203, 204, 205, 206, 208, 216, 217, 219,
                225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242,
                243, 245, 246, 247, 250, 252, 254, 256, 257, 258, 259, 260, 262, 264, 265, 266,
                268, 269, 270, 275, 277, 278, 279, 280, 282, 284, 285, 286, 287, 288, 290, 291,
                292, 294, 299, 301, 302, 303, 304, 306, 310, 311, 312, 314, 315, 316, 317, 318,
                319, 320, 321, 323, 328, 336, 337, 339, 341, 342, 343, 345, 349, 350, 352, 354,
                358, 359, 361, 365, 369, 422, 427, 442, 445, 454, 455, 456, 457, 458, 459, 463,
                464, 466, 467, 477, 478, 479, 480, 481, 482, 483, 484, 485, 487, 489, 490, 491,
                494, 495, 496, 498, 500, 501, 502, 503, 504, 506, 508, 509, 511, 512, 514, 516,
                519, 520, 521, 525, 526, 527, 534, 535, 536, 538, 539, 540, 542, 544, 545, 547,
                548, 549, 550, 551, 553, 554, 555, 557, 558, 560, 561, 562, 564, 565, 567, 568,
                570, 572, 573, 575, 576, 578, 579, 580, 581, 583, 584, 585, 586, 587, 589, 590,
                593, 595, 596, 597, 598, 599, 600, 602, 603, 604, 605, 606, 608, 609, 610, 611,
                612, 613, 616, 617, 618, 619, 620, 621, 624, 625, 626, 627, 628, 630, 631, 632,
                634, 638, 639, 640, 641, 645, 646, 649, 651, 652, 653, 656, 657, 659, 660, 661,
                662, 663, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681,
                682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 693, 694, 696, 697, 698, 700,
                704, 705, 706, 708, 710, 712, 713, 714, 715, 716, 717, 718, 719, 721, 722, 723,
                725, 728, 729, 732, 733, 737, 738, 740, 741, 744, 746, 747, 748, 750, 751, 752,
                753, 754, 756, 757, 759, 760, 762, 764, 765, 767, 768, 769, 771, 772, 773, 775,
                777, 779, 780, 781, 782, 783, 784, 785, 787, 791, 792, 796, 797, 798, 799, 800,
                801, 802, 803, 805, 806, 807, 809, 813, 814, 815, 817, 820, 821, 822, 823, 824,
                826
            ],
            "summary": {
                "covered_lines": 95,
                "missing_lines": 456,
                "percent_covered": 17.24137931034483
            }
        },
        "src/coverup/llm.py": {
            "executed_lines": [
                1, 2, 7, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 35,
                36, 38, 39, 41, 42, 47, 48, 51, 52, 54, 57, 71, 72
            ],
            "missing_lines": [
                49, 58, 60, 61, 63, 64, 65, 66, 68, 74, 76, 77, 78, 79, 81, 83, 84, 85, 87
            ],
            "summary": {
                "covered_lines": 35,
                "missing_lines": 19,
                "percent_covered": 64.81481481481481
            }
        },
        "src/coverup/segment.py": {
            "executed_lines": [
                1, 2, 3, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 31,
                34, 37, 60, 67, 68, 72
            ],
            "missing_lines": [
                27, 28, 32, 35, 38, 39, 40, 42, 43, 44, 46, 47, 48, 51, 52, 53, 55, 57, 61, 62, 64, 76,
                78, 80, 81, 83, 84, 85, 86, 88, 90, 91, 92, 95, 96, 97, 99, 100, 101, 103, 105, 106, 107,
                108, 109, 111, 113, 114, 115, 116, 119, 120, 121, 122, 123, 124, 126, 129, 131, 132, 133,
                134, 135, 136, 137, 138, 139, 141
            ],
            "summary": {
                "covered_lines": 29,
                "missing_lines": 68,
                "percent_covered": 29.896907216494846
            }
        },
        "src/coverup/utils.py": {
            "executed_lines": [ 1, 2, 3, 6, 29, 34, 52, 80 ],
            "missing_lines": [
                10, 11, 13, 14, 15, 16, 17, 19, 20, 22, 24, 26, 30, 31, 35, 37, 38, 39, 41, 42, 44, 45, 46,
                48, 49, 54, 56, 57, 59, 60, 61, 63, 65, 66, 67, 68, 69, 71, 72, 74, 75, 77, 82, 84
            ],
            "summary": {
                "covered_lines": 8,
                "missing_lines": 44,
                "percent_covered": 15.384615384615385
            }
        },
        "src/coverup/testrunner.py": {
            "executed_lines": [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 42, 74, 75, 78, 79, 80, 99, 167, 185 ],
            "missing_lines": [
                16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 31, 33, 34, 35, 36, 37, 39, 45, 46, 47, 48, 49, 50,
                51, 53, 54, 55, 56, 57, 59, 60, 61, 63, 64, 66, 68, 69, 70, 71, 81, 82, 84, 85, 86, 87, 90, 91,
                93, 94, 95, 96, 104, 106, 107, 109, 110, 112, 113, 115, 116, 118, 120, 121, 122, 123, 125, 126,
                128, 129, 130, 133, 134, 135, 136, 137, 138, 139, 140, 142, 144, 145, 146, 147, 149, 150, 151,
                153, 154, 157, 158, 159, 160, 161, 163, 164, 168, 169, 170, 172, 173, 174, 176, 177, 179, 180,
                182, 187, 189, 190, 191, 192, 193, 195, 196, 197, 198, 199, 201, 202, 203, 205, 206, 207, 208,
                210, 211, 213, 214, 216
            ],
            "summary": {
                "covered_lines": 21,
                "missing_lines": 130,
                "percent_covered": 13.907284768211921
            }
        },
        "src/coverup/delta.py": {
            "executed_lines": [ 1, 2, 5, 7, 9, 10, 11, 12, 13, 16, 18, 20, 21, 23, 24, 32, 35, 36, 39, 43, 44, 48 ],
            "missing_lines": [ 14, 25, 26, 27, 29, 30, 40, 49, 51, 52, 54, 56, 57, 58, 59, 61, 62, 63, 64, 67, 68 ],
            "summary": {
                "covered_lines": 22,
                "missing_lines": 21,
                "percent_covered": 51.16279069767442
            }
        },
        "src/coverup/prompt.py": {
            "executed_lines": [
                1, 2, 3, 4, 5, 8, 17, 18, 20, 25, 26, 30, 31, 35, 36, 40, 47, 48, 50, 54, 81, 90, 98, 99, 101, 105, 133,
                142, 149, 150, 152, 156, 195, 208, 221, 222, 223, 224
            ],
            "missing_lines": [
                10, 11, 12, 13, 14, 21, 22, 41, 42, 43, 51, 55, 56, 57, 58, 60, 61, 63, 64, 76, 82, 86, 91, 92, 102,
                106, 107, 108, 109, 111, 112, 114, 115, 127, 134, 138, 143, 144, 153, 157, 158, 159, 160, 162, 163, 164,
                165, 166, 167, 173, 196, 197, 209, 210
            ],
            "summary": {
                "covered_lines": 38,
                "missing_lines": 54,
                "percent_covered": 41.30434782608695
            }
        },
        "src/coverup/__main__.py": {
            "executed_lines": [],
            "missing_lines": [ 1, 3, 4 ],
            "summary": {
                "covered_lines": 0,
                "missing_lines": 3,
                "percent_covered": 0.0
            }
        },
        "src/coverup/pytest_plugin.py": {
            "executed_lines": [],
            "missing_lines": [
                1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 26, 28, 29, 31, 32, 33, 34,
                35, 37, 38, 39, 41, 42, 44, 45, 46, 48, 50, 51, 52, 54, 55, 56, 57, 59, 60, 61, 62, 64, 65, 67, 68, 69,
                70, 73, 74, 75, 78, 79, 80
            ],
            "summary": {
                "covered_lines": 0,
                "missing_lines": 60,
                "percent_covered": 0.0
            }
        }
    },
    "summary": {
        "covered_lines": 251,
        "missing_lines": 855,
        "percent_covered": 22.694394213381557
    }
}""")

    assert utils.summary_coverage(cov, []) == '22.7%'

    assert utils.summary_coverage(cov, [Path("src/coverup/prompt.py").resolve()]) == '41.3%'

    assert utils.summary_coverage(cov, [Path("src/coverup/prompt.py").resolve(),
                                        Path("src/coverup/pytest_plugin.py").resolve()]) == '25.0%'
