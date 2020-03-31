from abc import abstractmethod


class Agent(object):
  @abstractmethod
  def do_job(self):
	  pass

  @abstractmethod
  def get_action(self, state):
  	pass

  @abstractmethod
  def ready_to_play(self):
  	pass

