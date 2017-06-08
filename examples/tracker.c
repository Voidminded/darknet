#include "darknet.h"

void test_tracker()
{
  char *folders = "/local_home/autonomy_hands_temporal/list/list.txt";
  list *flist = get_paths(folders);
  char **folder_paths = (char **)list_to_array(flist);
  load_args args = {0};
  args.paths = folder_paths;
  int i;
  for( i=0; i < flist->size; ++i)
  {
    char *folder = folder_paths[i];
    list *frames = get_paths(folder);
    char **frame_paths = (char**)list_to_array(frames);
    strListMap *seq = malloc(sizeof(strListMap));
    seq->folder = folder;
    seq->frames = frame_paths;
    seq->count = frames->size;
    tsearch(seq, args.sequences, mapFind); /* insert */
  }
}
