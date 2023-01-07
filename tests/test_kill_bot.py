# python -m pytest tests/

from MCGladiator.baselines.kill_bot import KillBot


class TestKillBot:
    def tests_rotation(self):
        bot = KillBot("bot")

        # front right
        assert bot.compute_rotation((-2.7, 3.7), (-3.2, 0), -90)["camera"][1] < 0
        # front left
        assert bot.compute_rotation((-2.7, -2.7), (-3.2, 0), -90)["camera"][1] > 0
        # # back left
        assert bot.compute_rotation((13, -6), (-3.2, 0), 100)["camera"][1] < 0
        # # side right
        assert bot.compute_rotation((5.9, 17.5), (-3.2, 0), 145)["camera"][1] > 0
