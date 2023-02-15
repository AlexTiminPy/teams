import copy
import math
import random
import sys

import numpy
import pygame
import pickle

import warnings

from numba import njit

warnings.filterwarnings(action='ignore', category=UserWarning)

pygame.init()

with open('modelsSave/gaussModel.pkl', 'rb') as f:
    gaussModel = pickle.load(f)

xx0, xx1 = numpy.meshgrid(
    range(0, 1800, 10),
    range(0, 900, 100),
)

GRID = []

for i in zip(xx0, xx1):
    for t in zip(i[0], i[1]):
        GRID.append(list(t))


def get_models():
    with open('modelsSave/rotate_model.pkl', 'rb') as f:
        clf_rotate = pickle.load(file=f)

    with open('modelsSave/move_model.pkl', 'rb') as f:
        clf_move = pickle.load(file=f)

    with open('modelsSave/reload_pass_fire_model.pkl', 'rb') as f:
        clf_reload_pass_fire = pickle.load(file=f)

    return clf_rotate, clf_move, clf_reload_pass_fire


class Color:
    BLACK = (40, 40, 40)
    WHITE = (215, 215, 215)
    RED = (200, 50, 50)
    GREEN = (50, 200, 50)
    BLUE = (30, 67, 200)
    YELLOW = (200, 200, 50)
    GRAY = (125, 125, 125)
    List_color = ["BLUE", 'GREEN', "RED", "YELLOW", 'GRAY']
    gauss_colors = [(15, 33, 100), (25, 100, 25)]

    @staticmethod
    def random_color():
        return random.choice(Color.List_color)


class Circle:
    def __init__(self, color, x, y, radius):
        self.color = color
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self):
        pygame.draw.circle(win, self.color, [self.x, self.y], self.radius)


class Sector:
    def __init__(self, first_grad, second_grad, x, y, dx, dy):
        self.first_grad = first_grad
        self.second_grad = second_grad
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

    def draw(self):
        pygame.draw.arc(win, Color.RED, [self.x, self.y, self.dx, self.dy],
                        math.radians(-self.second_grad % 360),
                        math.radians(-self.first_grad % 360), 2)

        pygame.draw.line(win, Color.BLACK,
                         [self.x + self.dx / 2, self.y + self.dy / 2],
                         [(math.cos(math.radians(self.first_grad)) * (self.dx / 2)) + self.x + self.dx / 2,
                          (math.sin(math.radians(self.first_grad)) * (self.dy / 2)) + self.y + self.dy / 2], 2)

        pygame.draw.line(win, Color.BLACK,
                         [self.x + self.dx / 2, self.y + self.dy / 2],
                         [(math.cos(math.radians(self.second_grad)) * (self.dx / 2)) + self.x + self.dx / 2,
                          (math.sin(math.radians(self.second_grad)) * (self.dy / 2)) + self.y + self.dy / 2], 2)

        pygame.draw.line(win, Color.BLACK,
                         [self.x + self.dx / 2, self.y + self.dy / 2],
                         [(math.cos(math.radians(min(self.first_grad, self.second_grad) + 45)) * (self.dx / 2)) + self.x + self.dx / 2,
                          (math.sin(math.radians(min(self.first_grad, self.second_grad) + 45)) * (self.dy / 2)) + self.y + self.dy / 2], 2)


class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self):
        pygame.draw.line(win, Color.BLACK, [self.x1, self.y1], [self.x2, self.y2], 1)


class Patron:
    patrons = []

    def __init__(self, x: float, y: float, dx: float, dy: float, gun_damage: int,
                 fly_distance: float = 1500, radius: int = 1, speed: int = 500, color: Color = Color.BLACK
                 ):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

        self.gun_damage = gun_damage
        self.fly_distance = fly_distance
        self.distance = 0
        self.speed = speed

        self.radius = radius
        self.color = color

        self.is_alife = True

        Patron.patrons.append(self)

    def get_data_for_draw(self):
        return Circle(self.color, self.x, self.y, self.radius)

    def calculate_replace_position(self):
        self.distance += self.speed
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed


class Gun:
    def __init__(self, patron_count: int = 15, clip_cooldown: float = 360, patron_cooldown: int = 20,
                 patron_speed: int = 500, spread: float = 4, fire_distance: float = 500, damage: int = 5):
        self.max_patron_count = patron_count
        self.actual_patron_count = patron_count
        self.fire_speed = clip_cooldown
        self.patron_speed = patron_speed
        self.spread = spread
        self.fire_distance = fire_distance
        self.damage = damage

        self.patron_cooldown = patron_cooldown
        self.actual_patron_cooldown = patron_cooldown

        self.clip_cooldown = clip_cooldown
        self.actual_clip_cooldown = clip_cooldown
        self.is_reloaded = True


class ExternalPartWarrior:
    def __init__(self, x: float = 0, y: float = 0, radius: int = 3, color: Color = Color.random_color()):
        self.x = x
        self.y = y
        self.dx = 0
        self.dy = 0
        self.radius = radius
        self.color = color

        self.circle = Circle(self.color, x, y, self.radius)

    def get_data_for_draw(self):
        return [self.circle]

    def replace(self, dx, dy):
        self.x += dx
        self.y += dy
        self.circle.x += dx
        self.circle.y += dy


class InternalPartWarrior:
    def __init__(self):
        self.rotate_model = DecisionMakingWarriors.mutate(clf_rotate)
        self.move_model = DecisionMakingWarriors.mutate(clf_move)
        self.reload_pass_fire_model = DecisionMakingWarriors.mutate(clf_reload_pass_fire)


class FightPartWarrior:
    def __init__(self, patrons_count: int = 5000, heals: int = 1, speed: float = 0.5, rotation_speed: float = 1,
                 watch_angle: float = 90, watch_distance: int = 1500):
        self.max_patrons_count = patrons_count
        self.actual_patrons_count = patrons_count
        self.max_heals = heals
        self.actual_heals = heals
        self.speed = speed
        self.rotation_speed = rotation_speed

        self.watch_angle = watch_angle
        self.watch_distance = watch_distance
        self.actual_angle = random.randint(0, 360)


class Team:
    def __init__(self, name, color, id):
        self.name = name
        self.color = color
        self.id = id


class Warrior:
    warriors = []

    def __init__(self, gun: Gun, team: Team, external: ExternalPartWarrior, internal: InternalPartWarrior, fight: FightPartWarrior):
        self.gun = gun

        self.team = team

        self.external = external
        self.internal = internal
        self.fight = fight

        Warrior.warriors.append(self)

    def __tick__(self):
        if self.gun.actual_patron_cooldown < self.gun.patron_cooldown:
            self.gun.actual_patron_cooldown += 1

        if self.gun.actual_clip_cooldown < self.gun.clip_cooldown:
            self.gun.actual_clip_cooldown += 1

    def get_data_for_draw(self):

        return self.external.get_data_for_draw()

    def rotate(self, percent):
        self.fight.actual_angle += self.fight.rotation_speed * percent

    def went(self, percent_forward):
        dx = math.cos(math.radians(self.fight.actual_angle)) * (self.fight.speed * percent_forward)
        dy = math.sin(math.radians(self.fight.actual_angle)) * (self.fight.speed * percent_forward)
        if 0 < self.external.x + dx < WIDTH:
            self.external.replace(dx, 0)
        if 0 < self.external.y + dy < HEIGHT:
            self.external.replace(0, dy)
        pass

    def fire(self):

        if self.gun.actual_patron_cooldown != self.gun.patron_cooldown or self.gun.actual_clip_cooldown != self.gun.clip_cooldown:
            return

        if self.gun.actual_patron_count <= 0:
            return

        self.gun.actual_patron_cooldown = 0

        self.gun.actual_patron_count -= 1

        Patron(x=self.external.x + (math.cos(math.radians(self.fight.actual_angle)) * self.external.radius),
               y=self.external.y + (math.sin(math.radians(self.fight.actual_angle)) * self.external.radius),
               dx=math.cos(math.radians(self.fight.actual_angle)),
               dy=math.sin(math.radians(self.fight.actual_angle)),
               gun_damage=self.gun.damage,
               speed=self.gun.patron_speed)

    def reload(self):

        if self.gun.actual_clip_cooldown < self.gun.clip_cooldown:
            return

        if self.fight.actual_patrons_count <= 0:
            return

        self.gun.actual_clip_cooldown = 0
        self.gun.actual_patron_count += min(self.gun.max_patron_count, self.fight.actual_patrons_count)
        self.fight.actual_patrons_count -= min(self.gun.max_patron_count, self.fight.actual_patrons_count)


class DecisionMakingWarriors:
    @staticmethod
    def mutate(model):
        new_model = copy.deepcopy(model)
        for cf in new_model.coefs_:
            for i in range(len(cf)):
                for t in range(len(cf[i])):
                    cf[i][t] += random.uniform(-0.01, 0.01)

        return new_model

    @staticmethod
    def calculate_neural_network(warrior, all_possibles_enemy):
        start_data = DecisionMakingWarriors.get_start_data(warrior, all_possibles_enemy)
        rotate, move, reload_pass_fire = DecisionMakingWarriors.get_predicts(warrior, start_data)
        DecisionMakingWarriors.activate(warrior, rotate, move, reload_pass_fire)

    @staticmethod
    def get_start_data(warrior, all_possibles_enemy, is_print=False):
        look_at_friend = 0
        look_at_enemy = 0

        enemy_on_left = 0
        enemy_on_right = 0

        if is_print:
            print(len(all_possibles_enemy))

        filtered_to_angles_enemies = list(filter(lambda enemy: abs(DecisionMakingWarriors.get_angle(warrior, enemy)) < 45, all_possibles_enemy))
        if is_print:
            print(len(filtered_to_angles_enemies))

        without_left = list(filter(lambda enemy: not -45 < DecisionMakingWarriors.get_angle(warrior, enemy) < -5 and enemy.team is not warrior.team, filtered_to_angles_enemies))
        if len(without_left) < len(filtered_to_angles_enemies):
            enemy_on_right = True
        if is_print:
            print(len(without_left))

        without_right = list(filter(lambda enemy: not 45 > DecisionMakingWarriors.get_angle(warrior, enemy) > 5 and enemy.team is not warrior.team, without_left))
        if len(without_right) < len(without_left):
            enemy_on_left = True
        if is_print:
            print(len(without_right))

        sorted_to_distance_enemies = sorted(without_right, key=lambda enemy: abs(math.hypot(enemy.external.x - warrior.external.x, enemy.external.y - warrior.external.y)))
        if is_print:
            print("-" * 10)

        for enemy in sorted_to_distance_enemies:

            if enemy is warrior:
                continue

            look_at_enemy, look_at_friend = DecisionMakingWarriors.get_is_look_on(warrior, enemy)

            if look_at_enemy == 1 or look_at_friend == 1:
                break

        if sorted_to_distance_enemies:
            min_distance = abs(math.hypot(sorted_to_distance_enemies[0].external.x - warrior.external.x, sorted_to_distance_enemies[0].external.y - warrior.external.y))
        else:
            min_distance = 0

        if is_print:
            print(look_at_enemy,
                  look_at_friend,
                  int(bool(enemy_on_left)),
                  int(bool(enemy_on_right)))

        return [
            DecisionMakingWarriors.get_easy_data(warrior, min_distance) +
            [
                look_at_enemy,
                look_at_friend,
                int(bool(enemy_on_left)),
                int(bool(enemy_on_right))
            ]]

    @staticmethod
    def get_is_look_on(warrior, enemy):
        look_at_enemy = 0
        look_at_friend = 0

        if not collision_segment_and_circle(
                enemy.external.x, enemy.external.y, enemy.external.radius,
                warrior.external.x, warrior.external.y,
                warrior.external.x + math.cos(math.radians(warrior.fight.actual_angle)) * warrior.fight.watch_distance,
                warrior.external.y + math.sin(math.radians(warrior.fight.actual_angle)) * warrior.fight.watch_distance):
            return look_at_enemy, look_at_friend

        if enemy.team is warrior.team:
            look_at_friend = 1
            look_at_enemy = 0
            return look_at_enemy, look_at_friend

        look_at_enemy = 1
        look_at_friend = 0

        return look_at_enemy, look_at_friend

    @staticmethod
    def get_easy_data(warrior, min_distance):
        return [warrior.fight.actual_heals / warrior.fight.max_heals,
                warrior.gun.actual_patron_count / warrior.gun.max_patron_count,
                warrior.fight.actual_patrons_count / warrior.fight.max_patrons_count,
                (min_distance / warrior.fight.watch_distance)]

    @staticmethod
    def get_angle(warrior, enemy):
        angle = math.degrees(math.atan2(enemy.external.y - warrior.external.y, enemy.external.x - warrior.external.x))
        angle = max(angle, angle + 360)

        act_angle = max(warrior.fight.actual_angle, warrior.fight.actual_angle + 360)

        diff = act_angle - angle
        return diff

    @staticmethod
    def get_left_right_enemy(warrior, enemy, last_enemy_on_left, last_enemy_on_right):
        enemy_on_left = last_enemy_on_left
        enemy_on_right = last_enemy_on_right
        angle = math.degrees(math.atan2(enemy.external.y - warrior.external.y, enemy.external.x - warrior.external.x))
        angle = max(angle, angle + 360)

        act_angle = max(warrior.fight.actual_angle, warrior.fight.actual_angle + 360)

        if act_angle > angle > act_angle - 45:
            enemy_on_left += 1

        elif act_angle + 45 > angle > act_angle:
            enemy_on_right += 1

        return enemy_on_left, enemy_on_right

    @staticmethod
    def activate(warrior, rotate, move, reload_pass_fire):

        warrior.rotate(rotate[0])

        warrior.went(move[0])

        if reload_pass_fire < 0:
            warrior.reload()
        elif reload_pass_fire > 0:
            warrior.fire()

    @staticmethod
    def get_predicts(warrior, start_data):
        rotate = warrior.internal.rotate_model.predict(start_data)
        move = warrior.internal.move_model.predict(start_data)
        reload_pass_fire = warrior.internal.reload_pass_fire_model.predict(start_data)

        return rotate, move, reload_pass_fire


@njit
def collision_segment_and_segment(ax1, ay1, ax2, ay2,
                                  bx1, by1, bx2, by2):
    v1 = (bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)
    v2 = (bx2 - bx1) * (ay2 - by1) - (by2 - by1) * (ax2 - bx1)
    v3 = (ax2 - ax1) * (by1 - ay1) - (ay2 - ay1) * (bx1 - ax1)
    v4 = (ax2 - ax1) * (by2 - ay1) - (ay2 - ay1) * (bx2 - ax1)
    if (v1 * v2 < 0) and (v3 * v4 < 0):
        return True
    else:
        return False


@njit
def collision_segment_and_circle(x, y, radius,
                                 x1, y1,
                                 x2, y2):
    if abs(math.hypot(x - x1, y - y1)) > abs(math.hypot(x2 - x1, y2 - y1)):
        return False

    if radius > abs(math.hypot(x - x1, y - y1)):
        return True

    try:
        point = numpy.linalg.solve(numpy.array([[y2 - y1, x1 - x2],
                                                [x1 - x2, y1 - y2]]),
                                   numpy.array([(x2 - x1) * -y1 - (y2 - y1) * -x1,
                                                (y2 - y1) * -y - (x2 - x1) * x]))
    except:
        return False

    distance = math.hypot(point[0] - x, point[1] - y)

    if max(x1, x2) >= point[0] >= min(x1, x2) and \
            max(y1, y2) >= point[1] >= min(y1, y2):

        return distance <= radius

    else:
        return False


WIDTH, HEIGHT = 1800, 900
win = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()
fps = 120

my_font = pygame.font.SysFont('Comic Sans MS', 15)

clf_rotate, clf_move, clf_reload_pass_fire = get_models()

team1 = Team("Micky", Color.BLUE, id=0)
team2 = Team("Goffy", Color.GREEN, id=1)
# team3 = Team("user", Color.GRAY)

for i in range(5):
    Warrior(gun=Gun(),
            team=team1,
            external=ExternalPartWarrior(x=800, y=100 * i + 100, color=team1.color),
            internal=InternalPartWarrior(),
            fight=FightPartWarrior())
    Warrior(gun=Gun(),
            team=team2,
            external=ExternalPartWarrior(x=1000, y=100 * i + 100, color=team2.color),
            internal=InternalPartWarrior(),
            fight=FightPartWarrior())

spawn_count = 1


class BloodStain:
    stains = []

    def __init__(self, x, y, r):
        BloodStain.stains.append([x, y, r])

    @staticmethod
    def draw():
        for i in BloodStain.stains:
            if i[-1] <= 8:
                i[-1] += 0.1
            pygame.draw.circle(win, (150, 0, 0), i[0:2], i[-1])


def patr(patron):
    patron.calculate_replace_position()

    if patron.distance > patron.fly_distance:
        patron.is_alife = False

    patron.get_data_for_draw().draw()


def warr(warrior):
    warrior.__tick__()
    DecisionMakingWarriors.calculate_neural_network(warrior, Warrior.warriors)
    for drawable in warrior.get_data_for_draw():
        drawable.draw()


# user_warrior = Warrior(gun=Gun(),
#                        team=team3,
#                        external=ExternalPartWarrior(x=450, y=200, color=team3.color),
#                        internal=InternalPartWarrior(),
#                        fight=FightPartWarrior(watch_angle=0))

dataset = []

while True:
    pygame.display.set_caption(f"{spawn_count}, {len(Warrior.warriors)}, {CLOCK.get_fps()}, {CLOCK.get_time()}")
    # pygame.display.set_caption(f"{user_warrior.fight.actual_patrons_count}, {user_warrior.gun.actual_patron_count}")
    # pygame.display.set_caption(f"{min([warrior.fight.actual_patrons_count for warrior in Warrior.warriors])}")
    # dataset.append([len(Warrior.warriors), CLOCK.get_fps(), CLOCK.get_time()])

    win.fill(Color.WHITE)

    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    key = pygame.key.get_pressed()

    gaussModel.fit([[warrior.external.x, warrior.external.y] for warrior in Warrior.warriors], [warrior.team.id for warrior in Warrior.warriors])
    predict = gaussModel.predict(GRID)

    for grid, class_ in zip(GRID, predict):
        pygame.draw.rect(win, Color.gauss_colors[class_], grid + [100, 100])

    if len(Warrior.warriors) < 50:
        Warrior(gun=Gun(),
                team=team1,
                external=ExternalPartWarrior(x=random.randint(600, 800), y=random.randint(100, 800), color=team1.color),
                internal=InternalPartWarrior(),
                fight=FightPartWarrior(watch_angle=180))

        Warrior(gun=Gun(),
                team=team2,
                external=ExternalPartWarrior(x=random.randint(1000, 1200), y=random.randint(100, 800), color=team2.color),
                internal=InternalPartWarrior(),
                fight=FightPartWarrior(watch_angle=0))

    # if key[pygame.K_w]:
    #     user_warrior.went(1)
    #
    # if key[pygame.K_s]:
    #     user_warrior.went(-1)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            print(dataset)
            sys.exit()

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                print(dataset)
                sys.exit()

            if event.key == pygame.K_SPACE:
                Warrior.warriors = []

        if event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1:
                for i in range(spawn_count):
                    for t in range(spawn_count):
                        Warrior(gun=Gun(),
                                team=team1,
                                external=ExternalPartWarrior(x=mouse[0] + (25 * i - ((spawn_count - 1) * 25 / 2)),
                                                             y=mouse[1] + (25 * t - ((spawn_count - 1) * 25 / 2)), color=team1.color),
                                internal=InternalPartWarrior(),
                                fight=FightPartWarrior(watch_angle=0))

            if event.button == 3:
                for i in range(spawn_count):
                    for t in range(spawn_count):
                        Warrior(gun=Gun(),
                                team=team2,
                                external=ExternalPartWarrior(x=mouse[0] + (25 * i - ((spawn_count - 1) * 25 / 2)),
                                                             y=mouse[1] + (25 * t - ((spawn_count - 1) * 25 / 2)), color=team2.color),
                                internal=InternalPartWarrior(),
                                fight=FightPartWarrior(watch_angle=180))

            # if event.button == 1:
            #     user_warrior.fire()
            #
            # if event.button == 3:
            #     user_warrior.reload()

            if event.button == 4:
                spawn_count += 1

            if event.button == 5:
                spawn_count -= 1

    # user_warrior.fight.actual_angle = math.degrees(math.atan2(mouse[1] - user_warrior.external.y, mouse[0] - user_warrior.external.x))
    # DecisionMakingWarriors.get_start_data(user_warrior, Warrior.warriors, True)

    # BloodStain.draw()

    list(map(lambda warrior: warr(warrior), Warrior.warriors))

    patron_counter = 0

    while patron_counter < len(Patron.patrons):

        patron = Patron.patrons[patron_counter]

        warrior_counter = 0

        while warrior_counter < len(Warrior.warriors):

            warrior = Warrior.warriors[warrior_counter]

            if not collision_segment_and_circle(warrior.external.x, warrior.external.y, warrior.external.radius,
                                                patron.x, patron.y,
                                                patron.x + patron.dx * patron.speed,
                                                patron.y + patron.dy * patron.speed):
                warrior_counter += 1
                continue

            warrior.fight.actual_heals -= patron.gun_damage

            Patron.patrons.remove(patron)

            if warrior.fight.actual_heals <= 0:
                # BloodStain(warrior.external.x, warrior.external.y, random.randint(1, 3))
                Warrior.warriors.remove(warrior)

            warrior_counter += 1

            break

        patr(patron)
        patron_counter += 1

    # for i in user_warrior.get_data_for_draw():
    #     i.draw()
    #
    # Sector(user_warrior.fight.actual_angle - 45, user_warrior.fight.actual_angle + 45,
    #        user_warrior.external.x - user_warrior.fight.watch_distance, user_warrior.external.y - user_warrior.fight.watch_distance,
    #        user_warrior.fight.watch_distance * 2, user_warrior.fight.watch_distance * 2).draw()

    pygame.draw.circle(win, Color.BLACK, mouse, max((spawn_count - 1) * 25, 5), 5)

    pygame.display.flip()
    CLOCK.tick(fps)
