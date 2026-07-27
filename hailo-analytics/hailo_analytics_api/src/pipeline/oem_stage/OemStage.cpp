#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "OemStage.h"

int		OemStageInit(void)
{
	fprintf(stderr, "%s=%d\n", __FUNCTION__, __LINE__);
	return 0;
}

void	OemStageTerm(void)
{
	fprintf(stderr, "%s=%d\n", __FUNCTION__, __LINE__);
}

int		OemStageRun(void)
{
	fprintf(stderr, "%s=%d\n", __FUNCTION__, __LINE__);
	return 0;
}


