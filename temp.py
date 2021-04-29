import ast
import csv

baseline = """
STATIC_STATS: {'percent_solved': 0.161, 'cum_reward': 1241.6652961969376, 'max_possible_reward': 8000, 'reward_mean': 0.1552081620246172, 'reward_std': 0.35465082072108567, 'reward_solved_mean': 0.9640258510845788, 'reward_solved_std': 0.037512022879914425}
(4, 3, 0): {'percent_solved': 0.344375, 'cum_reward': 2666.1624885201454, 'max_possible_reward': 8000, 'reward_mean': 0.3332703110650182, 'reward_std': 0.46022736566619715, 'reward_solved_mean': 0.9677540793176571, 'reward_solved_std': 0.030874409700529484}
(2, 3, 4): {'percent_solved': 0.1855, 'cum_reward': 1452.6922951936722, 'max_possible_reward': 8000, 'reward_mean': 0.18158653689920903, 'reward_std': 0.380592518207591, 'reward_solved_mean': 0.9789031638771376, 'reward_solved_std': 0.016520442784837306}
(1, 6, 7): {'percent_solved': 0.3005, 'cum_reward': 2353.1442883610725, 'max_possible_reward': 8000, 'reward_mean': 0.2941430360451341, 'reward_std': 0.44892549070737076, 'reward_solved_mean': 0.9788453778540235, 'reward_solved_std': 0.01900380739807517}
(10, 11, 12): {'percent_solved': 0.243875, 'cum_reward': 1894.6344932317734, 'max_possible_reward': 8000, 'reward_mean': 0.23682931165397167, 'reward_std': 0.41727931948806596, 'reward_solved_mean': 0.9711094275918879, 'reward_solved_std': 0.028731171303456662}
"""

dynamic = """
STATIC_STATS: {'percent_solved': 0.373125, 'cum_reward': 2523.9068925604224, 'max_possible_reward': 8000, 'reward_mean': 0.3154883615700528, 'reward_std': 0.4280129310112043, 'reward_solved_mean': 0.8455299472564229, 'reward_solved_std': 0.20678969836155828}
(4, 3, 0): {'percent_solved': 0.289625, 'cum_reward': 2057.5471936762333, 'max_possible_reward': 8000, 'reward_mean': 0.25719339920952916, 'reward_std': 0.4112425737159036, 'reward_solved_mean': 0.8880220948106315, 'reward_solved_std': 0.1538715997828741}
(2, 3, 4): {'percent_solved': 0.3965, 'cum_reward': 2556.5238948464394, 'max_possible_reward': 8000, 'reward_mean': 0.31956548685580494, 'reward_std': 0.41333189400745307, 'reward_solved_mean': 0.8059659189301511, 'reward_solved_std': 0.19701402853745362}
(1, 6, 7): {'percent_solved': 0.034625, 'cum_reward': 192.74560081213713, 'max_possible_reward': 8000, 'reward_mean': 0.02409320010151714, 'reward_std': 0.13352205929393046, 'reward_solved_mean': 0.6958324939066323, 'reward_solved_std': 0.2181318959362527}
(10, 11, 12): {'percent_solved': 0.405875, 'cum_reward': 2844.038190692663, 'max_possible_reward': 8000, 'reward_mean': 0.3555047738365829, 'reward_std': 0.44338992872019745, 'reward_solved_mean': 0.8758971945465547, 'reward_solved_std': 0.16885001004763614}
"""

randomization = """
STATIC_STATS: {'percent_solved': 0.1955, 'cum_reward': 1500.7338960766792, 'max_possible_reward': 8000, 'reward_mean': 0.1875917370095849, 'reward_std': 0.38118222972526583, 'reward_solved_mean': 0.9595485269032475, 'reward_solved_std': 0.04899578418401395}
(4, 3, 0): {'percent_solved': 0.1645, 'cum_reward': 1251.4663962423801, 'max_possible_reward': 8000, 'reward_mean': 0.15643329953029753, 'reward_std': 0.35376734869416127, 'reward_solved_mean': 0.9509623071750609, 'reward_solved_std': 0.07169427472642916}
(2, 3, 4): {'percent_solved': 0.29575, 'cum_reward': 2208.7787910550833, 'max_possible_reward': 8000, 'reward_mean': 0.2760973488818854, 'reward_std': 0.4282670906055461, 'reward_solved_mean': 0.9335497848922584, 'reward_solved_std': 0.07951012189159075}
(1, 6, 7): {'percent_solved': 0.25725, 'cum_reward': 1943.232991375029, 'max_possible_reward': 8000, 'reward_mean': 0.24290412392187863, 'reward_std': 0.4142367246104676, 'reward_solved_mean': 0.9442337178693047, 'reward_solved_std': 0.0687343574912683}
(10, 11, 12): {'percent_solved': 0.252, 'cum_reward': 1884.0075900554657, 'max_possible_reward': 8000, 'reward_mean': 0.23550094875693323, 'reward_std': 0.4098520926608707, 'reward_solved_mean': 0.9345275744322746, 'reward_solved_std': 0.11509076298950838}
"""

static = """
STATIC_STATS: {'percent_solved': 0.365625, 'cum_reward': 2740.359987080097, 'max_possible_reward': 8000, 'reward_mean': 0.3425449983850121, 'reward_std': 0.4531276020458006, 'reward_solved_mean': 0.9368752092581529, 'reward_solved_std': 0.06848201725286875}
(4, 3, 0): {'percent_solved': 0.371125, 'cum_reward': 2837.231382280588, 'max_possible_reward': 8000, 'reward_mean': 0.3546539227850735, 'reward_std': 0.46260583320209275, 'reward_solved_mean': 0.9556185187876687, 'reward_solved_std': 0.04766671596165512}
(2, 3, 4): {'percent_solved': 0.108875, 'cum_reward': 839.915797829628, 'max_possible_reward': 8000, 'reward_mean': 0.1049894747287035, 'reward_std': 0.30080870016501055, 'reward_solved_mean': 0.9643120526172537, 'reward_solved_std': 0.04839846921366117}
(1, 6, 7): {'percent_solved': 0.19425, 'cum_reward': 1511.5775954425335, 'max_possible_reward': 8000, 'reward_mean': 0.18894719943031668, 'reward_std': 0.38513466410474045, 'reward_solved_mean': 0.9727011553684257, 'reward_solved_std': 0.033821118725506694}
(10, 11, 12): {'percent_solved': 0.028, 'cum_reward': 214.40959858894348, 'max_possible_reward': 8000, 'reward_mean': 0.026801199823617936, 'reward_std': 0.15855802228721724, 'reward_solved_mean': 0.9571857079863548, 'reward_solved_std': 0.0851446113063008}
"""

baseline_their_encoding = """
STATIC_STATS: {'percent_solved': 0.7325, 'cum_reward': 5624.416766732931, 'max_possible_reward': 8000, 'reward_mean': 0.7030520958416164, 'reward_std': 0.4267328727893538, 'reward_solved_mean': 0.9597980830602272, 'reward_solved_std': 0.04633526815788355}
(10, 11, 12): {'percent_solved': 0.233375, 'cum_reward': 1813.0560939311981, 'max_possible_reward': 8000, 'reward_mean': 0.22663201174139977, 'reward_std': 0.4110267734603267, 'reward_solved_mean': 0.9711066384205668, 'reward_solved_std': 0.029290158015224475}
(4, 3, 0): {'percent_solved': 0.33325, 'cum_reward': 2579.2942894548178, 'max_possible_reward': 8000, 'reward_mean': 0.32241178618185223, 'reward_std': 0.456524551291996, 'reward_solved_mean': 0.9674772278525198, 'reward_solved_std': 0.03516406343516797}
(2, 3, 4): {'percent_solved': 0.181625, 'cum_reward': 1422.1752955913544, 'max_possible_reward': 8000, 'reward_mean': 0.1777719119489193, 'reward_std': 0.37745214391952175, 'reward_solved_mean': 0.9787854752865481, 'reward_solved_std': 0.017344274907099947}
(1, 6, 7): {'percent_solved': 0.2985, 'cum_reward': 2338.6046884655952, 'max_possible_reward': 8000, 'reward_mean': 0.2923255860581994, 'reward_std': 0.44824644855708995, 'reward_solved_mean': 0.9793151961748724, 'reward_solved_std': 0.015867342918858957}
"""

csv_file = ""
heads = ""

def parse_file(file, agent_name):
  global csv_file, heads
  lines = file.split("\n")
  for line in lines:
    if line == "":
      continue
    splitty = line.split(": ")
    head = splitty[0]
    if head == "(4, 3, 0)":
      continue
    tail = ": ".join(splitty[1:])
    tail = ast.literal_eval(tail)

    if heads == "":
      heads = list(tail.keys())
      heads.append("shift")
      heads.append("agent")
    tail["shift"] = head
    tail["agent"] = agent_name

    with open("temp.txt", 'w') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=heads)
      writer.writeheader()
      writer.writerow(tail)
    with open("temp.txt", "r") as csvfile:
      content = csvfile.read()
      if csv_file == "":
        csv_file += content
      else:
        csv_file += content.split("\n")[1]+"\n"


for x,y  in zip([baseline, dynamic, randomization, static, baseline_their_encoding], "baseline, dynamic, randomization, static, baseline_their_encoding".split(", ")):
  parse_file(x, y)
print(csv_file)